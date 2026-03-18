#!/usr/bin/env python
"""
SHAP Analysis for MafAFP
========================
Two-level SHAP explanation for the GatedFiLM classifier:

  1. Classifier-level (1280-dim): joint PLM + MAF feature attribution
  2. MAF-only level  (128-dim):  MAF feature attribution with fixed PLM mean

Usage:
    python -m MafExtractor.predict_model.src.shap_analysis \
        --model_dir <path_to_saved_model> \
        --val_csv   <path_to_val_csv>
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from MafExtractor.predict_model.model.MafAFP.MafAFPmodel import (
    MafAFPClassifier,
)
from MafExtractor.maf.models.maf_extractor import MafExtractor
from MafExtractor.predict_model.utils.loadDataset import XYDataset_MAF
from MafExtractor.predict_model.src.MafAFP_train import collate_padding
from esm.sdk.api import ESMProtein
from MafExtractor.predict_model.utils.get_embedding import get_esm_embedding


# ==============================================================
# Wrapper Models for SHAP
# ==============================================================

class ClassifierWrapper(nn.Module):
    """Wraps GatedFiLMClassifier for classifier-level SHAP.

    Input:  [B, seq_dim + maf_dim]  (default 1152 + 128 = 1280)
    Output: [B, 1]  sigmoid probability
    """

    def __init__(self, classifier, seq_dim=1152):
        super().__init__()
        self.classifier = classifier
        self.seq_dim = seq_dim

    def forward(self, x):
        h_seq = x[:, :self.seq_dim]
        h_maf = x[:, self.seq_dim:]
        logits = self.classifier(h_seq, h_maf)
        return torch.sigmoid(logits)


class MAFOnlyWrapper(nn.Module):
    """Wraps GatedFiLMClassifier for MAF-only SHAP.

    Fixes h_seq to the dataset mean; only h_maf varies.
    Input:  [B, maf_dim]  (default 128)
    Output: [B, 1]  sigmoid probability
    """

    def __init__(self, classifier, h_seq_mean):
        super().__init__()
        self.classifier = classifier
        self.register_buffer("h_seq_mean", h_seq_mean)

    def forward(self, h_maf):
        B = h_maf.shape[0]
        h_seq = self.h_seq_mean.expand(B, -1)
        logits = self.classifier(h_seq, h_maf)
        return torch.sigmoid(logits)


class ModulatedFeatureWrapper(nn.Module):
    """Wraps the classification head for post-modulation SHAP.

    Input:  [B, hidden]  (default 512) — the post-LN fused representation z
    Output: [B, 1]  sigmoid probability
    """

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, z):
        return torch.sigmoid(self.head(z))


# ==============================================================
# Pre-compute Embeddings
# ==============================================================

def precompute_embeddings(model, maf_model, val_loader, device,
                          max_padding_length=100):
    """Run ESM-C backbone + MafExtractor on the validation set.

    Returns dict with **CPU** tensors:
        h_seq      [N, 1152]
        h_maf      [N, 128]
        labels     [N]
        sequences  list[str]
    """
    model.eval()
    maf_model.eval()

    all_h_seq, all_h_maf, all_labels, all_sequences = [], [], [], []

    with torch.no_grad():
        for datas in tqdm(val_loader, desc="Pre-computing embeddings"):
            text = datas["sequences"]
            labels = datas["labels"].float()
            maf_X = datas["maf_Xs"]

            # MafExtractor (runs on its own device, typically CPU)
            out = maf_model(maf_X, text)
            h_maf = out["global_feat"]

            # ESM-C backbone -> CLS embedding
            protein = ESMProtein(sequence=datas)
            protein_tensor = get_esm_embedding(protein, max_padding_length)
            with torch.amp.autocast("cuda"):
                h_seq = model.forward_backbone_only(
                    sequence_tokens=protein_tensor.sequence
                )
            # 取 CLS token -> [B, 1152]，与 h_maf [B, 128] 维度一致用于 SHAP 分析
            if h_seq.dim() == 3:
                h_seq = h_seq[:, 0, :]

            all_h_seq.append(h_seq.float().cpu())
            all_h_maf.append(h_maf.float().cpu())
            all_labels.append(labels.cpu())
            if isinstance(text, (list, tuple)):
                all_sequences.extend(text)
            else:
                all_sequences.append(text)

    return {
        "h_seq": torch.cat(all_h_seq, dim=0),
        "h_maf": torch.cat(all_h_maf, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "sequences": all_sequences,
    }


# ==============================================================
# Extract Post-Modulation Features
# ==============================================================

def extract_post_modulation_features(classifier, h_seq, h_maf, device):
    """Replicate GatedFiLMClassifier.forward() internals to extract
    intermediate representations.

    Parameters
    ----------
    classifier : GatedFiLMClassifier
    h_seq : Tensor [N, seq_dim] or [N, L, seq_dim]  (CPU)
    h_maf : Tensor [N, maf_dim]  (CPU)
    device : str or torch.device

    Returns
    -------
    dict with CPU tensors, each [N, hidden]:
        z_post      — post-LN fused representation (classification head input)
        z_seq       — projected PLM features before FiLM
        z_seq_film  — FiLM-modulated PLM features
        z_ca_contrib — cross-attention contribution: alpha * g_maf * z_maf_ca
    """
    classifier.eval()
    N = h_seq.shape[0]
    batch_size = 64

    all_z_post, all_z_seq, all_z_seq_film, all_z_ca_contrib = [], [], [], []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            hs = h_seq[start:end].to(device)
            hm = h_maf[start:end].to(device)

            # Clean NaN (same as forward)
            hs = torch.nan_to_num(hs, 0.0, 0.0, 0.0)
            hm = torch.nan_to_num(hm, 0.0, 0.0, 0.0)

            # 1) Projection
            q_seq, kv_seq = classifier._prep_seq_tokens(hs)  # [B,1,H], [B,L,H]
            kv_maf = classifier._tokenize_global(hm, classifier.maf_proj)  # [B,1,H]
            kv = torch.cat([kv_seq, kv_maf], dim=1)  # [B, L+1, H]

            # 2) Cross-attention
            z_maf_ca, _ = classifier.ca_maf(
                q_seq, kv, kv,
                need_weights=False, average_attn_weights=False,
            )  # [B,1,H]

            # 3) z_seq = q_seq (projected PLM, before FiLM)
            z_seq_b = q_seq  # [B,1,H]

            # 4) FiLM
            film_in = hm  # [B, maf_dim]
            gamma_beta = classifier.film(film_in)  # [B, 2H]
            gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B,H]
            gamma = torch.tanh(gamma) * classifier.film_gamma_scale
            z_seq_film_b = (1.0 + gamma).unsqueeze(1) * z_seq_b + beta.unsqueeze(1)  # [B,1,H]

            # 5) Gating + alpha
            g_maf = torch.sigmoid(
                classifier.gate_maf(hm) / classifier.gate_temp
            ).unsqueeze(1)  # [B,1,H]
            alpha_maf = torch.sigmoid(classifier.alpha_maf_logit)  # scalar

            # 6) Cross-attention contribution
            ca_contrib_b = alpha_maf * (g_maf * z_maf_ca)  # [B,1,H]

            # 7) Fusion + LayerNorm
            z = z_seq_film_b + ca_contrib_b  # [B,1,H]
            z = classifier.post_ln(z)  # [B,1,H]
            z = z.squeeze(1)  # [B,H]

            all_z_post.append(z.cpu())
            all_z_seq.append(z_seq_b.squeeze(1).cpu())
            all_z_seq_film.append(z_seq_film_b.squeeze(1).cpu())
            all_z_ca_contrib.append(ca_contrib_b.squeeze(1).cpu())

    return {
        "z_post": torch.cat(all_z_post, dim=0),
        "z_seq": torch.cat(all_z_seq, dim=0),
        "z_seq_film": torch.cat(all_z_seq_film, dim=0),
        "z_ca_contrib": torch.cat(all_z_ca_contrib, dim=0),
    }


# ==============================================================
# Classifier-level SHAP  (1280-dim)
# ==============================================================

def classifier_level_shap(classifier, h_seq, h_maf, labels,
                           device, output_dir, n_background=100):
    """SHAP on the fused [h_seq | h_maf] representation."""

    seq_dim = h_seq.shape[1]
    maf_dim = h_maf.shape[1]

    X = torch.cat([h_seq, h_maf], dim=1)          # [N, 1280]
    N = X.shape[0]

    feat_names = ([f"PLM_{i}" for i in range(seq_dim)]
                  + [f"MAF_{i}" for i in range(maf_dim)])

    wrapper = ClassifierWrapper(classifier, seq_dim=seq_dim).to(device).eval()

    n_bg = min(n_background, N)
    bg_idx = np.random.choice(N, size=n_bg, replace=False)
    background = X[bg_idx].to(device)

    print(f"[SHAP-Classifier] N={N}, features={X.shape[1]}, background={n_bg}")

    explainer = shap.GradientExplainer(wrapper, background)
    sv = explainer.shap_values(X.to(device))

    if isinstance(sv, list):
        sv = sv[0]
    if isinstance(sv, torch.Tensor):
        sv = sv.cpu().numpy()
    if sv.ndim == 3 and sv.shape[-1] == 1:
        sv = sv.squeeze(-1)

    X_np = X.numpy()
    labels_np = labels.numpy().ravel()

    # Base value = mean prediction on background samples
    with torch.no_grad():
        ev = float(wrapper(background).mean().cpu().item())

    # ---- (a) Summary beeswarm (top 30) ----
    plt.figure()
    shap.summary_plot(sv, X_np, feature_names=feat_names,
                      max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classifier_beeswarm_top30.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (b) Bar plot (mean |SHAP| top 30) ----
    plt.figure()
    shap.summary_plot(sv, X_np, feature_names=feat_names,
                      plot_type="bar", max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classifier_bar_top30.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (c) Grouped modality bar plot ----
    plm_importance = np.abs(sv[:, :seq_dim]).sum(axis=1).mean()
    maf_importance = np.abs(sv[:, seq_dim:]).sum(axis=1).mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["PLM (ESM-C)\n1152 dims", "MAF (Physics)\n128 dims"],
        [plm_importance, maf_importance],
        color=["#4C72B0", "#DD8452"],
        width=0.5,
    )
    ax.set_ylabel("Mean |SHAP| (summed over dimensions)")
    ax.set_title("Feature Modality Importance: PLM vs MAF")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.4f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classifier_modality_importance.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (d) Waterfall: top-1 AFP and top-1 non-AFP ----
    with torch.no_grad():
        probs = wrapper(X.to(device)).cpu().numpy().ravel()

    afp_mask = labels_np == 1
    non_afp_mask = labels_np == 0

    if afp_mask.any():
        top_afp_idx = int(np.where(afp_mask)[0][np.argmax(probs[afp_mask])])
        explanation_afp = shap.Explanation(
            values=sv[top_afp_idx],
            base_values=ev,
            data=X_np[top_afp_idx],
            feature_names=feat_names,
        )
        plt.figure()
        shap.plots.waterfall(explanation_afp, max_display=15, show=False)
        plt.title(f"Top-1 AFP (prob={probs[top_afp_idx]:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "classifier_waterfall_afp.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    if non_afp_mask.any():
        top_non_idx = int(np.where(non_afp_mask)[0][
            np.argmin(probs[non_afp_mask])
        ])
        explanation_non = shap.Explanation(
            values=sv[top_non_idx],
            base_values=ev,
            data=X_np[top_non_idx],
            feature_names=feat_names,
        )
        plt.figure()
        shap.plots.waterfall(explanation_non, max_display=15, show=False)
        plt.title(f"Top-1 Non-AFP (prob={probs[top_non_idx]:.3f})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "classifier_waterfall_non_afp.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    print(f"[SHAP-Classifier] Plots saved to {output_dir}")
    return sv, feat_names


# ==============================================================
# MAF-only SHAP  (128-dim)
# ==============================================================

def maf_only_shap(classifier, h_seq, h_maf, labels,
                   device, output_dir, n_background=100):
    """SHAP on h_maf only, fixing h_seq to its dataset mean."""

    N, maf_dim = h_maf.shape
    feat_names = [f"MAF_{i}" for i in range(maf_dim)]

    h_seq_mean = h_seq.mean(dim=0, keepdim=True)     # [1, 1152]
    wrapper = MAFOnlyWrapper(classifier, h_seq_mean.to(device))
    wrapper = wrapper.to(device).eval()

    n_bg = min(n_background, N)
    bg_idx = np.random.choice(N, size=n_bg, replace=False)
    background = h_maf[bg_idx].to(device)

    print(f"[SHAP-MAF] N={N}, features={maf_dim}, background={n_bg}")

    explainer = shap.GradientExplainer(wrapper, background)
    sv = explainer.shap_values(h_maf.to(device))

    if isinstance(sv, list):
        sv = sv[0]
    if isinstance(sv, torch.Tensor):
        sv = sv.cpu().numpy()
    if sv.ndim == 3 and sv.shape[-1] == 1:
        sv = sv.squeeze(-1)

    X_np = h_maf.numpy()
    labels_np = labels.numpy().ravel()

    # Base value = mean prediction on background samples
    with torch.no_grad():
        ev = float(wrapper(background).mean().cpu().item())

    mean_abs = np.abs(sv).mean(axis=0)

    # ---- (a) Summary beeswarm (all 128 dims) ----
    plt.figure()
    shap.summary_plot(sv, X_np, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maf_beeswarm.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (b) Bar plot (mean |SHAP|) ----
    plt.figure()
    shap.summary_plot(sv, X_np, feature_names=feat_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maf_bar.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (c) Dependence plot for top-3 MAF dims ----
    top3_idx = np.argsort(mean_abs)[::-1][:3]

    for rank, fidx in enumerate(top3_idx):
        plt.figure()
        shap.dependence_plot(
            fidx, sv, X_np, feature_names=feat_names, show=False,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir,
                f"maf_dependence_top{rank + 1}_{feat_names[fidx]}.png",
            ),
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    # ---- (d) Violin plot split by AFP / non-AFP ----
    afp_mask = labels_np == 1
    non_afp_mask = labels_np == 0
    top20_idx = np.argsort(mean_abs)[::-1][:20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, mask, title, color in [
        (axes[0], afp_mask, "AFP", "#DD8452"),
        (axes[1], non_afp_mask, "Non-AFP", "#4C72B0"),
    ]:
        if not mask.any():
            ax.set_title(f"{title} (no samples)")
            continue

        sv_sub = sv[mask][:, top20_idx]
        parts = ax.violinplot(
            sv_sub, positions=range(len(top20_idx)),
            showmeans=True, showextrema=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(top20_idx)))
        ax.set_xticklabels(
            [feat_names[i] for i in top20_idx], rotation=90, fontsize=8,
        )
        ax.set_title(f"SHAP Distribution \u2014 {title}")
        ax.set_ylabel("SHAP value")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    plt.suptitle("MAF Feature SHAP Distribution by Class (Top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maf_violin_by_class.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SHAP-MAF] Plots saved to {output_dir}")
    return sv, feat_names


# ==============================================================
# Modulated-feature SHAP  (512-dim)
# ==============================================================

def modulated_feature_shap(classifier, z_post, z_seq_film, z_ca_contrib,
                            labels, device, output_dir, n_background=100):
    """SHAP on the post-modulation fused representation z_post (512-dim).

    Also generates a decomposition chart showing FiLM vs cross-attention
    contributions for the top SHAP dimensions.
    """
    N, hidden = z_post.shape
    feat_names = [f"Z_{i}" for i in range(hidden)]

    wrapper = ModulatedFeatureWrapper(classifier.head).to(device).eval()
    # Enable gradients on the head for SHAP
    for p in wrapper.parameters():
        p.requires_grad = True

    n_bg = min(n_background, N)
    bg_idx = np.random.choice(N, size=n_bg, replace=False)
    background = z_post[bg_idx].to(device)

    print(f"[SHAP-Modulated] N={N}, features={hidden}, background={n_bg}")

    explainer = shap.GradientExplainer(wrapper, background)
    sv = explainer.shap_values(z_post.to(device))

    if isinstance(sv, list):
        sv = sv[0]
    if isinstance(sv, torch.Tensor):
        sv = sv.cpu().numpy()
    if sv.ndim == 3 and sv.shape[-1] == 1:
        sv = sv.squeeze(-1)

    X_np = z_post.numpy()
    labels_np = labels.numpy().ravel()

    # Base value
    with torch.no_grad():
        ev = float(wrapper(background).mean().cpu().item())

    mean_abs = np.abs(sv).mean(axis=0)  # [hidden]
    top30_idx = np.argsort(mean_abs)[::-1][:30]

    # ---- (a) Beeswarm top 30 ----
    plt.figure()
    shap.summary_plot(sv, X_np, feature_names=feat_names,
                      max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "modulated_beeswarm_top30.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (b) Bar top 30 ----
    plt.figure()
    shap.summary_plot(sv, X_np, feature_names=feat_names,
                      plot_type="bar", max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "modulated_bar_top30.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (c) Decomposition: FiLM vs Cross-Attention contribution (top 30) ----
    film_contrib_np = z_seq_film.numpy()   # [N, hidden]
    ca_contrib_np = z_ca_contrib.numpy()   # [N, hidden]

    film_mean_abs = np.abs(film_contrib_np).mean(axis=0)  # [hidden]
    ca_mean_abs = np.abs(ca_contrib_np).mean(axis=0)      # [hidden]

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(top30_idx))
    dim_labels = [feat_names[i] for i in top30_idx]

    bars_film = ax.barh(y_pos, film_mean_abs[top30_idx],
                        label="FiLM (modulated PLM)", color="#4C72B0")
    bars_ca = ax.barh(y_pos, ca_mean_abs[top30_idx],
                      left=film_mean_abs[top30_idx],
                      label="Cross-Attention (gated MAF)", color="#DD8452")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dim_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |Contribution|")
    ax.set_title("Post-Modulation Feature Decomposition: FiLM vs Cross-Attention (Top 30 SHAP dims)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "modulated_decomposition_top30.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (d) Waterfall: top-1 AFP ----
    with torch.no_grad():
        probs = wrapper(z_post.to(device)).cpu().numpy().ravel()

    afp_mask = labels_np == 1
    non_afp_mask = labels_np == 0

    if afp_mask.any():
        top_afp_idx = int(np.where(afp_mask)[0][np.argmax(probs[afp_mask])])
        explanation_afp = shap.Explanation(
            values=sv[top_afp_idx],
            base_values=ev,
            data=X_np[top_afp_idx],
            feature_names=feat_names,
        )
        plt.figure()
        shap.plots.waterfall(explanation_afp, max_display=15, show=False)
        plt.title(f"Modulated Features — Top-1 AFP (prob={probs[top_afp_idx]:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "modulated_waterfall_afp.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # ---- (e) Waterfall: top-1 non-AFP ----
    if non_afp_mask.any():
        top_non_idx = int(np.where(non_afp_mask)[0][np.argmin(probs[non_afp_mask])])
        explanation_non = shap.Explanation(
            values=sv[top_non_idx],
            base_values=ev,
            data=X_np[top_non_idx],
            feature_names=feat_names,
        )
        plt.figure()
        shap.plots.waterfall(explanation_non, max_display=15, show=False)
        plt.title(f"Modulated Features — Top-1 Non-AFP (prob={probs[top_non_idx]:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "modulated_waterfall_non_afp.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    print(f"[SHAP-Modulated] Plots saved to {output_dir}")
    return sv, feat_names


# ==============================================================
# FiLM Effect Analysis  (non-SHAP)
# ==============================================================

def film_effect_analysis(z_seq, z_seq_film, labels, output_dir):
    """Analyze FiLM modulation effect by comparing z_seq vs z_seq_film.

    Generates 4 plots characterizing how FiLM transforms the PLM representation.
    """
    z_seq_np = z_seq.numpy()        # [N, hidden]
    z_film_np = z_seq_film.numpy()  # [N, hidden]
    labels_np = labels.numpy().ravel()
    delta = z_film_np - z_seq_np    # [N, hidden]
    hidden = delta.shape[1]

    afp_mask = labels_np == 1
    non_afp_mask = labels_np == 0

    mean_abs_delta = np.abs(delta).mean(axis=0)  # [hidden]
    top30_idx = np.argsort(mean_abs_delta)[::-1][:30]
    top10_idx = np.argsort(mean_abs_delta)[::-1][:10]

    # ---- (a) Heatmap: sample × dimension FiLM delta (top 30) ----
    fig, ax = plt.subplots(figsize=(14, 8))
    # Sort samples: AFP first, then non-AFP
    sorted_idx = np.concatenate([np.where(afp_mask)[0], np.where(non_afp_mask)[0]])
    delta_sub = delta[sorted_idx][:, top30_idx]

    vmax = np.percentile(np.abs(delta_sub), 95)
    im = sns.heatmap(delta_sub, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                     xticklabels=[f"Z_{i}" for i in top30_idx],
                     yticklabels=False, ax=ax,
                     cbar_kws={"label": "FiLM Δ (z_film − z_seq)"})
    # Mark class boundary
    n_afp = afp_mask.sum()
    if n_afp > 0 and non_afp_mask.any():
        ax.axhline(y=n_afp, color="black", linewidth=1.5, linestyle="--")
        ax.text(len(top30_idx) + 0.5, n_afp / 2, "AFP",
                va="center", ha="left", fontsize=10, fontweight="bold")
        ax.text(len(top30_idx) + 0.5, n_afp + non_afp_mask.sum() / 2, "Non-AFP",
                va="center", ha="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Samples (AFP | Non-AFP)")
    ax.set_title("FiLM Modulation Delta Heatmap (Top 30 Dimensions)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "film_delta_heatmap_top30.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (b) Violin: AFP vs non-AFP delta distribution (top 10) ----
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    axes = axes.flatten()
    for rank, dim_idx in enumerate(top10_idx):
        ax = axes[rank]
        data_afp = delta[afp_mask, dim_idx] if afp_mask.any() else np.array([])
        data_non = delta[non_afp_mask, dim_idx] if non_afp_mask.any() else np.array([])

        parts_list = []
        labels_list = []
        colors = []
        if len(data_afp) > 0:
            parts_list.append(data_afp)
            labels_list.append("AFP")
            colors.append("#DD8452")
        if len(data_non) > 0:
            parts_list.append(data_non)
            labels_list.append("Non-AFP")
            colors.append("#4C72B0")

        if parts_list:
            parts = ax.violinplot(parts_list, positions=range(len(parts_list)),
                                  showmeans=True, showextrema=True)
            for pc, c in zip(parts["bodies"], colors):
                pc.set_facecolor(c)
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(labels_list)))
            ax.set_xticklabels(labels_list, fontsize=8)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"Z_{dim_idx}", fontsize=9)
        if rank % 5 == 0:
            ax.set_ylabel("FiLM Δ")

    plt.suptitle("FiLM Delta Distribution: AFP vs Non-AFP (Top 10 Dimensions)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "film_delta_distribution_top10.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (c) Bar: all 512 dims mean|delta| ----
    fig, ax = plt.subplots(figsize=(14, 4))
    sorted_all = np.argsort(mean_abs_delta)[::-1]
    ax.bar(range(hidden), mean_abs_delta[sorted_all], color="#55A868", width=1.0)
    ax.set_xlabel("Dimension (sorted by mean |Δ|)")
    ax.set_ylabel("Mean |FiLM Δ|")
    ax.set_title(f"FiLM Effect Magnitude Across All {hidden} Dimensions")
    # Mark top-10 threshold
    if hidden > 10:
        ax.axvline(x=9.5, color="red", linewidth=0.8, linestyle="--",
                   label="Top-10 threshold")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "film_effect_magnitude.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- (d) Bar: class-discriminative dims (Cohen's d, top 20) ----
    cohens_d = np.zeros(hidden)
    if afp_mask.sum() >= 2 and non_afp_mask.sum() >= 2:
        delta_afp = delta[afp_mask]      # [n_afp, hidden]
        delta_non = delta[non_afp_mask]  # [n_non, hidden]
        mean_afp = delta_afp.mean(axis=0)
        mean_non = delta_non.mean(axis=0)
        std_afp = delta_afp.std(axis=0, ddof=1)
        std_non = delta_non.std(axis=0, ddof=1)
        n1, n2 = afp_mask.sum(), non_afp_mask.sum()
        pooled_std = np.sqrt(((n1 - 1) * std_afp**2 + (n2 - 1) * std_non**2) / (n1 + n2 - 2))
        pooled_std = np.where(pooled_std < 1e-8, 1e-8, pooled_std)
        cohens_d = (mean_afp - mean_non) / pooled_std

    top20_d_idx = np.argsort(np.abs(cohens_d))[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top20_d_idx))
    colors = ["#DD8452" if cohens_d[i] > 0 else "#4C72B0" for i in top20_d_idx]
    ax.barh(y_pos, cohens_d[top20_d_idx], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Z_{i}" for i in top20_d_idx], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Cohen's d (AFP − Non-AFP)")
    ax.set_title("FiLM Delta: Most Class-Discriminative Dimensions (Top 20)")
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#DD8452", label="AFP > Non-AFP"),
                       Patch(color="#4C72B0", label="Non-AFP > AFP")],
              loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "film_class_discriminative_dims.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[FiLM-Effect] Plots saved to {output_dir}")


# ==============================================================
# Main pipeline
# ==============================================================

def run_shap_pipeline(model_dir, val_csv, maf_checkpoint=None,
                      device="cuda", n_background=100, output_dir=None):
    """Full SHAP analysis pipeline.

    Parameters
    ----------
    model_dir : str
        Directory containing ``best_AUC.pkl``.
    val_csv : str
        Path to validation CSV (columns: sequence, label).
    maf_checkpoint : str or None
        Path to MafExtractor checkpoint.  Auto-detected if *None*.
    device : str
        Torch device  (default ``"cuda"``).
    n_background : int
        Number of background samples for GradientExplainer.
    output_dir : str or None
        Where to write plots / npz.  Defaults to ``<model_dir>/shap/``.
    """
    if output_dir is None:
        output_dir = os.path.join(model_dir, "shap")
    os.makedirs(output_dir, exist_ok=True)

    root_path = os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ))
    )

    # ---- Load MafExtractor (kept on CPU, same as training) ----
    if maf_checkpoint is None:
        maf_checkpoint = os.path.join(
            root_path, "MafExtractor", "maf", "output", "g_if", "best_maf.pt"
        )

    maf_model = MafExtractor()
    maf_model.load_state_dict(
        torch.load(maf_checkpoint, map_location="cpu", weights_only=True)
    )
    maf_model.eval()
    for p in maf_model.parameters():
        p.requires_grad = False

    # ---- Load MafAFPClassifier ----
    model_args = {
        "model_name": "esmc_600m",
        "classifier_hidden_ratio": 1,
        "id2label": {"pos": 1, "neg": 0},
        "use_flash_attention": True,
    }
    model = MafAFPClassifier.from_esm_pretrained(**model_args)

    ckpt_path = os.path.join(model_dir, "best_AUC.pkl")
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # ---- Validation DataLoader ----
    val_df = pd.read_csv(val_csv)
    val_dataset = XYDataset_MAF(val_df)
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=0, collate_fn=collate_padding,
    )

    # ---- Step 1: Pre-compute embeddings ----
    print("=" * 60)
    print("Step 1: Pre-computing embeddings")
    print("=" * 60)
    emb = precompute_embeddings(model, maf_model, val_loader, device)
    h_seq = emb["h_seq"]
    h_maf = emb["h_maf"]
    labels = emb["labels"]
    print(f"  h_seq: {h_seq.shape},  h_maf: {h_maf.shape},  "
          f"labels: {labels.shape}")

    # Enable gradients on the classifier for SHAP
    classifier = model.classifier
    for p in classifier.parameters():
        p.requires_grad = True

    # ---- Step 2: Classifier-level SHAP (1280-dim) ----
    print("=" * 60)
    print("Step 2: Classifier-level SHAP (1280-dim)")
    print("=" * 60)
    sv_clf, names_clf = classifier_level_shap(
        classifier, h_seq, h_maf, labels, device, output_dir, n_background,
    )

    # ---- Step 3: MAF-only SHAP (128-dim) ----
    print("=" * 60)
    print("Step 3: MAF-only SHAP (128-dim)")
    print("=" * 60)
    sv_maf, names_maf = maf_only_shap(
        classifier, h_seq, h_maf, labels, device, output_dir, n_background,
    )

    # ---- Step 4: Extract post-modulation features ----
    print("=" * 60)
    print("Step 4: Extracting post-modulation features")
    print("=" * 60)
    mod_feats = extract_post_modulation_features(
        classifier, h_seq, h_maf, device,
    )
    z_post = mod_feats["z_post"]
    z_seq_internal = mod_feats["z_seq"]
    z_seq_film = mod_feats["z_seq_film"]
    z_ca_contrib = mod_feats["z_ca_contrib"]
    print(f"  z_post: {z_post.shape},  z_seq: {z_seq_internal.shape},  "
          f"z_seq_film: {z_seq_film.shape},  z_ca_contrib: {z_ca_contrib.shape}")

    # ---- Step 5: Modulated-feature SHAP (512-dim) ----
    print("=" * 60)
    print("Step 5: Modulated-feature SHAP (512-dim)")
    print("=" * 60)
    sv_mod, names_mod = modulated_feature_shap(
        classifier, z_post, z_seq_film, z_ca_contrib,
        labels, device, output_dir, n_background,
    )

    # ---- Step 5b: FiLM effect analysis ----
    print("=" * 60)
    print("Step 5b: FiLM effect analysis")
    print("=" * 60)
    film_effect_analysis(z_seq_internal, z_seq_film, labels, output_dir)

    # ---- Step 6: Save SHAP values ----
    print("=" * 60)
    print("Step 6: Saving SHAP values")
    print("=" * 60)
    npz_path = os.path.join(output_dir, "shap_values.npz")
    np.savez(
        npz_path,
        classifier_shap=sv_clf,
        classifier_features=np.array(names_clf),
        maf_shap=sv_maf,
        maf_features=np.array(names_maf),
        modulated_shap=sv_mod,
        modulated_features=np.array(names_mod),
        h_seq=h_seq.numpy(),
        h_maf=h_maf.numpy(),
        labels=labels.numpy(),
        z_post=z_post.numpy(),
        z_seq=z_seq_internal.numpy(),
        z_seq_film=z_seq_film.numpy(),
        z_ca_contrib=z_ca_contrib.numpy(),
    )
    print(f"  SHAP values -> {npz_path}")
    print(f"  All plots   -> {output_dir}")

    return sv_clf, sv_maf, sv_mod


# ==============================================================
# CLI
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP Analysis for MafAFP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directory containing best_AUC.pkl",
    )
    parser.add_argument(
        "--val_csv", type=str, required=True,
        help="Path to validation CSV (columns: sequence, label)",
    )
    parser.add_argument(
        "--maf_checkpoint", type=str, default=None,
        help="Path to MafExtractor checkpoint (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for plots (default: <model_dir>/shap/)",
    )
    parser.add_argument(
        "--n_background", type=int, default=100,
        help="Number of background samples for SHAP",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_shap_pipeline(
        model_dir=args.model_dir,
        val_csv=args.val_csv,
        maf_checkpoint=args.maf_checkpoint,
        device=args.device,
        n_background=args.n_background,
        output_dir=args.output_dir,
    )
