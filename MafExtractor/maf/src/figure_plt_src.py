import argparse, os, ast
from torch.utils.data import DataLoader
from MafExtractor.maf.data.datasets import MafWindowDataset, collate_fn, BucketingSampler
from MafExtractor.maf.utils.aa import onehot_from_seqs,onehot_from_seqs_token,seq_to_tensor
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from pathlib import Path
from MafExtractor.maf.utils.aa import onehot_from_seqs
from MafExtractor.maf.models.maf_extractor import MafExtractor

import json


def moving_average(x, k=3):

    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k)/k, mode='same')


def plot_maf_single_sequence(
        seq: str,
        maf_model,
        output_dir,
        device="cuda",
        figsize=(16, 12),
        cmap="viridis",
        smooth_k=3,   # 平滑强度
        save_json_report=True
):


    maf_model.eval()
    maf_model.to(device)


    x = seq_to_tensor(seq).unsqueeze(0).to(device)

    seqs = [seq]


    with torch.no_grad():
        out = maf_model.forward_features(x, seqs)

    cls_scores = out["local_scores"]
    g_feat = out["global_feat"]


    onehot = onehot_from_seqs(seqs).to(device)
    ww_list = maf_model.ww_conv(onehot)
    hm_list = maf_model.hm_conv(onehot)

    ww_list = [t[0,0].cpu().numpy() for t in ww_list]   # [L_w]
    hm_list = [t[0,0].cpu().numpy() for t in hm_list]


    ww_list_smooth = [moving_average(w, smooth_k) for w in ww_list]
    hm_list_smooth = [moving_average(h, smooth_k) for h in hm_list]


    best_windows = []
    for w, ww in zip(maf_model.window_sizes, ww_list):
        idx = int(np.argmin(ww))
        best_windows.append((w, idx))


    L = len(seq)
    fig = plt.figure(figsize=figsize)


    ax1 = plt.subplot2grid((4,1), (0,0))

    heat = np.zeros((len(cls_scores), max(s.shape[1] for s in cls_scores)))
    for i, s in enumerate(cls_scores):
        arr = s.squeeze(0).cpu().numpy()
        # 归一化 / Clip 让显示更清晰
        arr = np.clip(arr, np.percentile(arr, 5), np.percentile(arr, 95))
        heat[i, :len(arr)] = arr

    sns.heatmap(heat, ax=ax1, cmap=cmap)
    ax1.set_title("MAF multi-scale cls_scores (CNN)",
                  fontsize=17, fontweight="bold")

    ax1.set_yticks(np.arange(len(cls_scores)) + 0.5)
    ax1.set_yticklabels(maf_model.window_sizes)


    ax1.set_xticks(np.arange(L) + 0.5)
    ax1.set_xticklabels([c for c in seq], fontsize=6, rotation=90)
    ax1.tick_params(labelsize=14)


    ax2 = plt.subplot2grid((4,1), (1,0))

    for (w, ww, hm) in zip(maf_model.window_sizes, ww_list_smooth, hm_list_smooth):
        ax2.plot(ww, label=f"ΔG_if (w={w})")
        ax2.plot(hm, '--', label=f"HM (w={w})")

    ax2.set_title("Sliding-window ΔG_if & HM",
                  fontsize=17, fontweight="bold")
    ax2.legend()
    ax2.tick_params(labelsize=14)


    ax3 = plt.subplot2grid((4,1), (2,0))

    ax3.plot(np.zeros(L))

    for w, idx in best_windows:
        ax3.axvspan(idx, idx + w, alpha=0.3, label=f"best w={w}")
        ax3.scatter([idx + w/2], [0], s=40, color="blue")

    ax3.set_title("Best window positions",
                  fontsize=17, fontweight="bold")
    ax3.legend()
    ax3.tick_params(labelsize=14)

    ax4 = plt.subplot2grid((4,1), (3,0))

    centers = []
    for w, idx in best_windows:
        centers.append(idx + w/2)

    ax4.plot(centers, marker="o")
    ax4.set_title("Best ΔG_if window center vs scale",
                  fontsize=17, fontweight="bold")

    ax4.set_xticks(range(len(maf_model.window_sizes)))
    ax4.set_xticklabels([f"w={w}" for w in maf_model.window_sizes])
    ax4.tick_params(labelsize=14)
    plt.tight_layout()

    save_path = Path(output_dir) / f"maf_{seq[:10]}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    if save_json_report:
        report = {
            "sequence": seq,
            "best_windows": [
                {"window_size": int(w), "start": int(idx), "end": int(idx+w)}
                for w, idx in best_windows
            ],
            "hydrophobic_core_positions": centers,
            "comment": "best_windows 代表最容易插入膜界面的片段（ΔG_if最低）"
        }

        with open(Path(output_dir) / f"maf_{seq[:10]}.json", "w") as f:
            json.dump(report, f, indent=4)

    return save_path

def extract_maf_features(seq, maf_model, device="cuda"):
    """复用：提取一个序列的 cls_scores / ΔG_if / HM / best_window 信息"""
    maf_model.eval()

    x = seq_to_tensor(seq).unsqueeze(0).to(device)
    seqs = [seq]

    with torch.no_grad():
        out = maf_model.forward_features(x, seqs)

    cls_scores = [s.squeeze(0).cpu().numpy() for s in out["local_scores"]]


    onehot = onehot_from_seqs(seqs).to(device)
    ww_list = maf_model.ww_conv(onehot)
    hm_list = maf_model.hm_conv(onehot)

    ww = [w.squeeze().cpu().numpy() for w in ww_list]
    hm = [h.squeeze().cpu().numpy() for h in hm_list]

    # best positions
    best_pos = []
    for w_size, ww_curve in zip(maf_model.window_sizes, ww):
        idx = int(np.argmin(ww_curve))
        best_pos.append((w_size, idx))

    return cls_scores, ww, hm, best_pos

def plot_maf_compare_sequences(
        seq_afp: str,
        seq_non_afp: str,
        maf_model,
        output_file,
        device="cuda",
        figsize=(18, 12),
        cmap="viridis"
):


    maf_model = maf_model.to(device).eval()


    afp_cls, afp_ww, afp_hm, afp_best = extract_maf_features(seq_afp, maf_model, device)
    non_cls, non_ww, non_hm, non_best = extract_maf_features(seq_non_afp, maf_model, device)

    window_sizes = maf_model.window_sizes
    max_w = max([len(r) for r in afp_cls + non_cls])


    all_vals = np.concatenate([np.concatenate(afp_cls), np.concatenate(non_cls)])
    vmin, vmax = np.min(all_vals), np.max(all_vals)


    fig, axes = plt.subplots(3, 2, figsize=figsize)


    ax = axes[0,0]
    heat_afp = np.zeros((len(window_sizes), max_w))
    for i, r in enumerate(afp_cls):
        heat_afp[i, :len(r)] = r
    sns.heatmap(heat_afp, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("AFP cls_scores heatmap",
                  fontsize=17, fontweight="bold")
    ax.set_yticks(np.arange(len(window_sizes)) + 0.5)
    ax.set_yticklabels(window_sizes)

    ax = axes[0,1]
    heat_non = np.zeros((len(window_sizes), max_w))
    for i, r in enumerate(non_cls):
        heat_non[i, :len(r)] = r
    sns.heatmap(heat_non, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Non-AFP cls_scores heatmap",
                  fontsize=17, fontweight="bold")
    ax.set_yticks(np.arange(len(window_sizes)) + 0.5)
    ax.set_yticklabels(window_sizes)

    ax_afp = axes[1,0]
    for (w, ww, hm) in zip(window_sizes, afp_ww, afp_hm):
        ax_afp.plot(ww, label=f"ΔG_if w={w}")
        ax_afp.plot(hm, linestyle="--", label=f"HM w={w}")
    ax_afp.set_title("AFP ΔG_if & HM curves",
                  fontsize=17, fontweight="bold")
    ax_afp.tick_params(labelsize=14)
    ax_non = axes[1,1]
    for (w, ww, hm) in zip(window_sizes, non_ww, non_hm):
        ax_non.plot(ww, label=f"ΔG_if w={w}")
        ax_non.plot(hm, linestyle="--", label=f"HM w={w}")
    ax_non.set_title("Non-AFP ΔG_if & HM curves",
                  fontsize=17, fontweight="bold")
    ax_non.tick_params(labelsize=14)

    ax_afp_b = axes[2,0]
    L_afp = len(seq_afp)
    ax_afp_b.plot(np.zeros(L_afp))
    for w, idx in afp_best:
        ax_afp_b.axvspan(idx, idx+w, alpha=0.3, label=f"{w}")
    ax_afp_b.set_title("AFP Best-window positions",
                  fontsize=17, fontweight="bold")
    ax_afp_b.tick_params(labelsize=14)
    ax_non_b = axes[2,1]
    L_non = len(seq_non_afp)
    ax_non_b.plot(np.zeros(L_non))
    for w, idx in non_best:
        ax_non_b.axvspan(idx, idx+w, alpha=0.3, label=f"{w}")
    ax_non_b.set_title("Non-AFP Best-window positions",
                  fontsize=17, fontweight="bold")
    ax_non_b.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_file}/{seq_afp[:5]}_vs_{seq_non_afp[:5]}.png", dpi=300)
    plt.close()

    print(f"[Saved] AFP vs Non-AFP 对比图 -> {output_file}")

def pad_to_max_length(arr_list):

    max_len = max(len(a) for a in arr_list)
    padded = []
    for a in arr_list:
        pad = np.full((max_len,), np.nan)
        pad[:len(a)] = a
        padded.append(pad)
    return np.vstack(padded)

def global_maf_interpretation(
        cfg,
        tmp_csv,
        amp_csv,
        maf_model,
        output_dir,
        device="cuda",
        figsize=(17, 12),
        cmap="viridis"
):

    device = cfg['train']['device'] if torch.cuda.is_available() and cfg['train']['device'] == 'cuda' else 'cpu'

    ds_te = MafWindowDataset(tmp_csv, amp_csv, cfg['model']['window_sizes'],
                             cfg['labels']['weak_pos_threshold'], cfg['labels']['weak_neg_threshold'],
                             cfg['data']['max_len'], split='test', split_ratio=cfg['data']['train_split'],
                             seed=cfg['seed'])

    lengths_te = [len(s.seq) for s in ds_te.samples]
    sampler_te = BucketingSampler(lengths_te, cfg['train']['batch_size'], shuffle=False, bucket_size=200)
    dl_te = DataLoader(ds_te, batch_sampler=sampler_te,  collate_fn=collate_fn)

    maf_model.eval()
    maf_model.to(device)


    window_sizes = maf_model.window_sizes

    all_cls = [[] for _ in window_sizes]
    all_ww  = [[] for _ in window_sizes]
    all_hm  = [[] for _ in window_sizes]
    all_best_pos  = [[] for _ in window_sizes]

    # ====== 逐序列累计 ======
    for seqs, X, labs, amp_labels in dl_te:
        X = X.to(device)


        with torch.no_grad():
            out = maf_model.forward_features(X, seqs)

        cls_list = out["local_scores"]  # list of [B, Lw]


        for i, s in enumerate(cls_list):
            s = s.cpu().numpy()   # [B, Lw]
            for row in s:
                all_cls[i].append(row)


        onehot = onehot_from_seqs_token(seqs).to(device)
        ww_list = maf_model.ww_conv(onehot)
        hm_list = maf_model.hm_conv(onehot)

        for i, (ww, hm) in enumerate(zip(ww_list, hm_list)):
            ww = ww.squeeze(1).cpu().numpy()   # [B, Lw]
            hm = hm.squeeze(1).cpu().numpy()
            for w1, h1 in zip(ww, hm):
                all_ww[i].append(w1)
                all_hm[i].append(h1)


        B = len(seqs)
        for b in range(B):
            seq = seqs[b]
            L = len(seq)


            onehot_s = onehot_from_seqs_token([seq]).to(device)
            ww_s = maf_model.ww_conv(onehot_s)

            for i, (w, w_tensor) in enumerate(zip(maf_model.window_sizes, ww_s)):
                arr = w_tensor.squeeze().cpu().numpy()  # [Lw]
                if len(arr) == 0:
                    continue
                idx = np.argmin(arr)
                center = idx + (w - 1) / 2
                center_norm = center / L
                all_best_pos[i].append(center_norm)
    avg_cls = []
    avg_ww  = []
    avg_hm  = []

    for i in range(len(window_sizes)):
        cls_pad = pad_to_max_length(all_cls[i])
        ww_pad = pad_to_max_length(all_ww[i])
        hm_pad = pad_to_max_length(all_hm[i])

        avg_cls.append(np.nanmean(cls_pad, axis=0))
        avg_ww.append(np.nanmean(ww_pad, axis=0))
        avg_hm.append(np.nanmean(hm_pad, axis=0))


    fig = plt.figure(figsize=figsize)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))


    for w, y in zip(window_sizes, avg_cls):
        axes[0].plot(y, label=f"cls w={w}")
    axes[0].set_title("Average cls_scores (multi-scale)", fontsize=17, fontweight="bold")
    axes[0].legend(fontsize=14)
    axes[0].tick_params(labelsize=14)


    for w, ww, hm in zip(window_sizes, avg_ww, avg_hm):
        axes[1].plot(ww, label=f"WW w={w}")
        axes[1].plot(hm, "--", label=f"HM w={w}")
    axes[1].set_title("Average WW & HM (multi-scale)", fontsize=17, fontweight="bold")
    axes[1].legend(fontsize=14)  # 图例多的话可以分两列
    axes[1].tick_params(labelsize=14)


    for i, w in enumerate(window_sizes):
        if len(all_best_pos[i]) == 0:
            continue
        axes[2].hist(all_best_pos[i], bins=40, alpha=0.5, label=f"w={w}")
    axes[2].set_title("Distribution of best-window positions (normalized)", fontsize=17, fontweight="bold")
    axes[2].set_xlabel("Relative position along sequence", fontsize=17, fontweight="bold")

    axes[2].legend()
    axes[2].legend(fontsize=14)
    axes[2].tick_params(labelsize=14)

    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / "global_maf_interpretation_afp.png", dpi=350)
    plt.close()

    print("Global Interpretation Saved:", Path(output_dir) / "global_maf_interpretation_afp.png")

# ------------------------- Demo -------------------------
if __name__ == '__main__':
    import numpy as np
    import torch
    #
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maf_path = f"{root_path}/maf/output/g_if/best_maf.pt"
    output_dir = f"{root_path}/maf/output/"
    #
    os.makedirs(output_dir, exist_ok=True)

    seq = "SLSVEAKAKIVADFGRDANDTGSSEVQVALLTAQINHLQGHFSEHKKDHHSRRGLLRMVSTRRKLLDYLKRKDVASYVSLIERLGLRR"
    non_afp = 'MWPKDENFEQRSSLECAQKAAELRASIKEKVELIRLKKLLHERNASLVMTKAQLTEVQEAYETLLQKNQGILSAAHEALLKQVNELRA'
    # non_afp = "GSNGHIGFNEPGTPFSSKTHVVELAEETRKANARYFPTLEDVP"
    maf = MafExtractor()
    maf.load_state_dict(torch.load(maf_path))
    # #
    # plot_maf_single_sequence(seq, maf, output_dir)
    # plot_maf_compare_sequences(seq,non_afp, maf, output_dir)

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default=f"{root_path}\maf\config.yaml")
    parser.add_argument('--tmp_csv', type=str, required=False, default=f'{root_path}\dataset/TMP/afp1.csv')
    parser.add_argument('--amp_csv', type=str, required=False, default=f'{root_path}\dataset/TMP/afp2.csv')
    parser.add_argument('--out_dir', type=str, required=False,default=f"{root_path}\maf\output")
    args = parser.parse_args()



    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            import yaml
            cfg = yaml.safe_load(f)


    global_maf_interpretation(cfg, args.tmp_csv, args.amp_csv, maf,output_dir)


