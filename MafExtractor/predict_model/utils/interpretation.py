# esmc/utils/interpretation.py
import torch
from esm.sdk.api import ESMProtein
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from pathlib import Path
from MafExtractor.predict_model.utils.get_embedding import get_esm_embedding


class InterpretationHook:
    """注册 Hook 捕获 cross-attn, FiLM γ/β, gate, α"""
    def __init__(self, model):
        self.model = model
        self.records = {}
        self.handles = []

    def _hook_cross_attn(self, module, input, output):
        q, k, v = input
        if hasattr(module, "last_attn"):
            self.records["cross_attn_weights"] = module.last_attn.detach().cpu()

    def _hook_film(self, module, input, output):
        h = output.detach().cpu()
        H = h.shape[-1] // 2
        self.records["film_gamma"] = h[:, :H]
        self.records["film_beta"] = h[:, H:]

    def _hook_gate(self, module, input, output):
        self.records["gate_output"] = torch.sigmoid(output.detach().cpu())

    def register(self):
        """注册 Hook 到模型关键模块"""
        if hasattr(self.model, "ca_maf"):
            self.handles.append(self.model.ca_maf.register_forward_hook(self._hook_cross_attn))
        if hasattr(self.model, "film"):
            self.handles.append(self.model.film.register_forward_hook(self._hook_film))
        if hasattr(self.model, "gate_maf"):
            self.handles.append(self.model.gate_maf.register_forward_hook(self._hook_gate))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def get_records(self):
        out = self.records.copy()
        self.records.clear()
        return out


# ============================================================
# 可视化函数
# ============================================================
def plot_cross_attention(attn_map, seq_tokens, maf_features, save_path):
    """绘制 Cross-Attention 热图"""
    attn = attn_map.detach().cpu()

    # 自动平均多余维度 (batch/head)
    while attn.dim() > 2:
        attn = attn.mean(0)

    attn = attn.squeeze()  # 去掉多余维度

    # 检查维度正确性
    if attn.dim() != 2:
        raise ValueError(f"Attention map must be 2D, got shape {attn.shape}")

    L_seq, L_maf = attn.shape
    seq_tokens = seq_tokens[:L_seq]
    maf_features = maf_features[:L_maf]

    attn_np = attn.numpy()

    plt.figure(figsize=(max(6, L_maf / 5), max(4, L_seq / 5)))
    sns.heatmap(attn_np, cmap="YlGnBu",
                xticklabels=maf_features,
                yticklabels=seq_tokens)
    plt.title("Cross-Attention (Seq × MAF)")
    plt.xlabel("MAF Attributes")
    plt.ylabel("Sequence")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
def plot_global_cross_attention(attn_map, seq_tokens, maf_features, save_path, title=None):
    """绘制 Cross-Attention 热图（兼容全局注意力）"""
    if attn_map is None or attn_map.numel() == 0:
        print(f"[Warn] Empty attention map, skip plot: {save_path}")
        return

    attn = attn_map.detach().cpu()
    attn_mean = attn.mean().item()  # 全局注意力强度

    plt.figure(figsize=(5, 4))
    sns.barplot(x=["Global Cross-Attn"], y=[attn_mean], color="skyblue")
    plt.ylabel("Attention Strength")
    plt.title(title or "Global Cross-Attention Strength (mean)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_film_gamma_beta(gamma, beta, save_prefix):
    """绘制 FiLM γ/β 参数变化"""
    gamma_mean = gamma.mean(0).numpy()
    beta_mean = beta.mean(0).numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(gamma_mean, label="γ (scale)")
    plt.plot(beta_mean, label="β (shift)")
    plt.legend()
    plt.title("FiLM γ/β Mean across Features")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_film_gamma_beta.png", dpi=300)
    plt.close()


def plot_gate_alpha(gate, alpha_logit, epoch, save_path):
    """绘制 Gate + α"""
    g_mean = gate.mean().item()
    alpha = torch.sigmoid(alpha_logit).item() if alpha_logit is not None else 0
    plt.figure(figsize=(6, 4))
    plt.bar(["Gate(MAF)", "Alpha(Residual)"], [g_mean, alpha])
    plt.title(f"MAF Contribution — Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 主解释函数
# ============================================================
def run_interpretation(model, seq_tensor, maf_tensor,
                       seq_labels=None,
                       maf_features=None,
                       output_dir="./interpretation",
                       epoch=None):
    """
    自动生成解释性可视化结果。
    model: EsmcClassifier
    seq_tensor: [B, L, D]
    maf_tensor: [B, D']
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    hooker = InterpretationHook(model.classifier)
    hooker.register()
    with torch.no_grad():
        out = model.forward_backbone_only(sequence_tokens=seq_tensor)
        y_pred = model.forward_classifier_only(out, maf_tensor)


    rec = hooker.get_records()
    hooker.remove()

    seq_labels = seq_labels or [f"AA{i}" for i in range(seq_tensor.size(1))]
    maf_features = maf_features or [f"MAF{i}" for i in range(maf_tensor.size(1))]

    # Cross-Attention
    if "cross_attn_weights" in rec:
        plot_cross_attention(rec["cross_attn_weights"][0], seq_labels, maf_features,
                             Path(output_dir) / f"epoch_{epoch}_cross_attn.png")

    # FiLM γ/β
    if "film_gamma" in rec and "film_beta" in rec:
        plot_film_gamma_beta(rec["film_gamma"], rec["film_beta"],
                             str(Path(output_dir) / f"epoch_{epoch}"))

    # Gate + α
    if "gate_output" in rec:
        plot_gate_alpha(rec["gate_output"],
                        getattr(model.classifier, "alpha_maf_logit", None),
                        epoch,
                        Path(output_dir) / f"epoch_{epoch}_gate_alpha.png")

    print(f"[Interpretation] Epoch {epoch} — 可视化结果已保存至: {output_dir}")

def run_global_interpretation(model, maf_model,val_loader,
                              maf_features=None, seq_len=None,
                              output_dir="./global_interpretation",
                              device="cuda", max_batches=None):
    """
    聚合整个验证集的 cross-attn / FiLM / Gate 特征并绘制全局解释图。
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    hooker = InterpretationHook(model.classifier)
    hooker.register()
    maf = maf_model.eval()
    print("[GlobalInterpret] 开始聚合验证集注意力/调制特征...")
    rec_all = {"cross_attn_weights": [], "film_gamma": [], "film_beta": [], "gate_output": []}
    with torch.no_grad():
        for i, datas in enumerate(tqdm(val_loader, desc="GlobalInterpret")):
            text = datas['sequences']
            y = datas['labels'].float().to('cuda')
            y = y.unsqueeze(-1)
            with torch.no_grad():
                maf_X = datas['maf_Xs']
                out = maf(maf_X, text)
                maf_feature = out["global_feat"].to('cuda')
                protein = ESMProtein(sequence=datas)
                protein_tensor = get_esm_embedding(protein, 100)
                out = model.forward_backbone_only(sequence_tokens=protein_tensor.sequence)
                y_pred = model.forward_classifier_only(out, maf_feature)
            rec = hooker.get_records()
            for k in rec_all.keys():
                if k in rec:
                    rec_all[k].append(rec[k])
    hooker.remove()

    if rec_all["cross_attn_weights"]:
        attn_all = torch.cat(rec_all["cross_attn_weights"], dim=0)
        attn_mean = attn_all.mean(0)
        plot_global_cross_attention(attn_mean,
                             [f"AA{i}" for i in range(attn_mean.shape[-2])],
                             maf_features or [f"MAF{i}" for i in range(attn_mean.shape[-1])],
                             Path(output_dir) / "global_cross_attn.png")

    if rec_all["film_gamma"] and rec_all["film_beta"]:
        gamma = torch.cat(rec_all["film_gamma"], dim=0)
        beta = torch.cat(rec_all["film_beta"], dim=0)
        plot_film_gamma_beta(gamma, beta,
                             str(Path(output_dir) / "global"))

    if rec_all["gate_output"]:
        gate = torch.cat(rec_all["gate_output"], dim=0)
        plt.figure(figsize=(6, 4))
        plt.hist(gate.numpy().flatten(), bins=40, color="skyblue")
        plt.title("Global Gate(MAF) Distribution")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "global_gate_hist.png", dpi=300)
        plt.close()

    print(f"[GlobalInterpret] 全局解释结果已保存至: {output_dir}")
