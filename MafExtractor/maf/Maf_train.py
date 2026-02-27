import argparse, os, ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from MafExtractor.maf.utils.common import set_seed, load_yaml_text, save_json
from MafExtractor.maf.models.maf_extractor import MafExtractor
from MafExtractor.maf.data.datasets import MafWindowDataset, collate_fn, BucketingSampler
from tqdm import tqdm
from typing import List, Dict
import torch
import torch.nn as nn

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os



class EarlyStoppingForMAF:



    def __init__(self, loss_window=5, loss_threshold=0.60, loss_fluct=0.02,
    local_mse_thr=0.05, feat_drift_thr=0.10, corr_thr=0.10,
        ckpt_path="best_maf_feature.pt"):
        self.loss_window = loss_window
        self.loss_threshold = loss_threshold
        self.loss_fluct = loss_fluct
        self.local_mse_thr = local_mse_thr
        self.feat_drift_thr = feat_drift_thr
        self.corr_thr = corr_thr
        self.ckpt_path = ckpt_path


        self.eval_losses = []
        self.prev_local = None
        self.prev_feat = None
        self.prev_feat_epoch = None


# ---------- 1. Eval loss ----------
    def check_loss_stable(self):
        if len(self.eval_losses) < self.loss_window:
            return False
        recent = self.eval_losses[-self.loss_window:]
        if min(recent) < self.loss_threshold and (max(recent) - min(recent) < self.loss_fluct):
            return True
        return False


    # ---------- 2. local_scores ----------
    def check_local_stability(self, local_scores: List[torch.Tensor]):
    # local_scores ：[B,Lw]
        if self.prev_local is None:
            self.prev_local = [t.detach().cpu() for t in local_scores]
            return False
        diffs = []
        for a, b in zip(self.prev_local, local_scores):
            a = a.to(b.device)
            diffs.append((a - b).abs().mean().item())
        avg_diff = sum(diffs) / len(diffs)
        self.prev_local = [t.detach().cpu() for t in local_scores]
        return avg_diff < self.local_mse_thr

    def check_feature_drift(self, g_feat: torch.Tensor, epoch: int):
        if self.prev_feat is None:
            self.prev_feat = g_feat.detach().cpu()

            self.prev_feat_epoch = epoch
            return False
        if epoch - self.prev_feat_epoch < 5:
            return False
        drift = torch.norm(g_feat.detach().cpu() - self.prev_feat, p=2, dim=1).mean().item()
        self.prev_feat = g_feat.detach().cpu()
        self.prev_feat_epoch = epoch
        return drift < self.feat_drift_thr


    @staticmethod
    def check_pca_corr(g_feat: torch.Tensor, labels: torch.Tensor, corr_thr=0.10):
        g = g_feat.detach().cpu().numpy()

        y = labels.detach().cpu().numpy()
        if len(np.unique(y)) == 1:
            return False
        pc1 = PCA(n_components=1).fit_transform(g)[:, 0]
        corr, _ = pearsonr(pc1, y)
        return abs(corr) > corr_thr

    def should_stop(self, eval_loss: float, local_scores, g_feat, labels, epoch, model):
        self.eval_losses.append(eval_loss)


        cond1 = self.check_loss_stable()
        cond2 = self.check_local_stability(local_scores)
        cond3 = self.check_feature_drift(g_feat, epoch)
        cond4 = self.check_pca_corr(g_feat, labels, self.corr_thr)


        print(f"[EarlyStop] loss:{cond1} local:{cond2} drift:{cond3} corr:{cond4}")


        if cond1 and cond2 and cond3 and cond4:
            print("[EarlyStop] 特征稳定性达到要求，停止训练。")
            torch.save(model.state_dict(),  os.path.join(self.ckpt_path, 'early_stop_maf.pt'))
            print(f"[EarlyStop] 已保存特征收敛 checkpoint: {self.ckpt_path}")
            return True
        return False


def visualize_local_scores(local_scores: List[torch.Tensor], seq: str, save_path=None):
    plt.figure(figsize=(10, 3 * len(local_scores)))
    for i, s in enumerate(local_scores):
        plt.subplot(len(local_scores), 1, i+1)
        arr = s[0].detach().cpu().numpy()
        plt.plot(arr)
        plt.title(f"Scale {i} window size")
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


# ---------- 2. g_maf PCA ----------
def visualize_maf_pca(g_feat: torch.Tensor, labels: torch.Tensor, save_path=None):
    X = g_feat.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    pc = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(6,6))
    plt.scatter(pc[:,0], pc[:,1], c=y, cmap="coolwarm", alpha=0.7)
    plt.title("g_maf PCA")
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def train_loop(cfg, tmp_csv, amp_csv, out_dir):
    set_seed(cfg['seed'])
    device = cfg['train']['device'] if torch.cuda.is_available() and cfg['train']['device'] == 'cuda' else 'cpu'

    ds_tr = MafWindowDataset(tmp_csv, amp_csv, cfg['model']['window_sizes'],
                             cfg['labels']['weak_pos_threshold'], cfg['labels']['weak_neg_threshold'],
                             cfg['data']['max_len'], split='train', split_ratio=cfg['data']['train_split'],
                             seed=cfg['seed'])
    ds_te = MafWindowDataset(tmp_csv, amp_csv, cfg['model']['window_sizes'],
                             cfg['labels']['weak_pos_threshold'], cfg['labels']['weak_neg_threshold'],
                             cfg['data']['max_len'], split='test', split_ratio=cfg['data']['train_split'],
                             seed=cfg['seed'])

    lengths_tr = [len(s.seq) for s in ds_tr.samples]
    sampler_tr = BucketingSampler(lengths_tr, cfg['train']['batch_size'], shuffle=True, bucket_size=200)
    dl_tr = DataLoader(ds_tr, batch_sampler=sampler_tr,  collate_fn=collate_fn)
    lengths_te = [len(s.seq) for s in ds_te.samples]
    sampler_te = BucketingSampler(lengths_te, cfg['train']['batch_size'], shuffle=False, bucket_size=200)
    dl_te = DataLoader(ds_te, batch_sampler=sampler_te,  collate_fn=collate_fn)


    model = MafExtractor(aa_dim=cfg['model']['aa_dim'], hidden=cfg['model']['hidden'],
    window_sizes=cfg['model']['window_sizes'], learnable_scales=cfg['model']['learnable_scales'],
    dropout=cfg['model']['dropout']).to(device)


    optim = torch.optim.AdamW(model.parameters(), lr=ast.literal_eval(cfg['train']['lr']), weight_decay=ast.literal_eval(cfg['train']['weight_decay']))
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()


    os.makedirs(out_dir, exist_ok=True)
    logs = {"train": [], "test": []}


    global_step = 0
    best_loss = 1e9

    earlystop = EarlyStoppingForMAF(
        loss_window=5,
        loss_threshold=0.60,
        loss_fluct=0.02,
        local_mse_thr=0.05,
        feat_drift_thr=0.10,
        corr_thr=0.10,
        ckpt_path=out_dir
    )

    for epoch in range(cfg['train']['epochs']):
        model.train()
        for seqs, X, labs,amp_labels in tqdm(dl_tr):
            X = X.to(device)
            out = model(X,seqs)
            # 计算窗口级损失（对齐不同尺度的标签）
            loss = 0.0
            for i,(s_pred, s_reg) in enumerate(zip(out['local_scores'], out['reg_scores'])):
                maxL = s_pred.size(1)
                Y = torch.zeros_like(s_pred)
                W = torch.zeros_like(s_pred)
                for bi,s_lab in enumerate(labs):
                    y = s_lab[i]
                    if y.numel()==0: continue
                    t = y.to(device)
                    Lw = min(maxL, t.size(0))
                    Y[bi,:Lw] = (t[:Lw] == 1).float()
                    W[bi,:Lw] = (t[:Lw] != -1).float()
                # BCEWithLogitsLoss 手动加权
                l_cls = (bce(s_pred, Y) * (W.mean()+1e-8)) # 近似权重
                # 回归头：没有真实 ΔG_if，使用 self-supervised 目标：让 s_reg ≈ s_pred（温和约束）
                l_reg = mse(s_reg, s_pred.detach())
                loss = loss + l_cls + 0.1*l_reg
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            optim.step()


            global_step += 1
            # if global_step % cfg['log']['eval_every'] == 0:
            #     logs['train'].append({"step": global_step, "loss": float(loss.item())})
                # print(f"[Train] step {global_step} loss={loss.item():.4f}")



        te_loss,last_local_scores,all_labels,all_gfeat = evaluate(model, dl_te, bce, mse, device)
        logs['test'].append({"epoch": epoch, "loss": te_loss})
        print(f"[Eval] epoch {epoch} loss={te_loss:.4f}")

        if te_loss < best_loss:
            best_loss = te_loss
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_maf.pt'))
        if (epoch + 1) % cfg['log']['save_every'] == 0:
            save_json(logs, os.path.join(out_dir, 'logs.json'))

        if earlystop.should_stop(
                te_loss,
                last_local_scores,
                all_gfeat,
                all_labels,
                epoch,
                model
        ):
            print(f"✅ Early stopped at epoch {epoch}")
            save_json(logs, os.path.join(out_dir, 'logs.json'))


    save_json(logs, os.path.join(out_dir, 'logs.json'))


def evaluate(model, dl, bce, mse, device):
    model.eval()
    eval_losses = []
    all_labels = []
    all_gfeat = []
    last_local_scores = None
    with torch.no_grad():
        for seqs, X, labs,amp_labels in tqdm(dl):
            X = X.to(device)
            out = model(X,seqs)
            loss = 0.0
            for i,(s_pred, s_reg) in enumerate(zip(out['local_scores'], out['reg_scores'])):
                maxL = s_pred.size(1)
                Y = torch.zeros_like(s_pred)
                W = torch.zeros_like(s_pred)
                for bi,s_lab in enumerate(labs):
                    y = s_lab[i]
                    if y.numel()==0: continue
                    t = y.to(device)
                    Lw = min(maxL, t.size(0))
                    Y[bi,:Lw] = (t[:Lw] == 1).float()
                    W[bi,:Lw] = (t[:Lw] != -1).float()
                l_cls = (bce(s_pred, Y) * (W.mean()+1e-8))
                l_reg = mse(s_reg, s_pred.detach())
                loss = loss + l_cls + 0.1*l_reg
            eval_losses.append(loss.item())
            last_local_scores = out['local_scores']
            all_gfeat.append(out['global_feat'])
            all_labels.append(torch.tensor(amp_labels))
    te_loss = np.mean(eval_losses)
    all_gfeat = torch.cat(all_gfeat, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return te_loss,last_local_scores,all_labels,all_gfeat


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default=f"{root_path}\maf\config.yaml")
    parser.add_argument('--tmp_csv', type=str, required=False, default=f'{root_path}\dataset\TMP/topdb_output_regions.csv')
    parser.add_argument('--amp_csv', type=str, required=False, default=f'{root_path}\dataset\cpl_diff/train/antifungal_label.csv')
    parser.add_argument('--out_dir', type=str, required=False,default=f"{root_path}\maf\output")
    args = parser.parse_args()



    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            import yaml
            cfg = yaml.safe_load(f)


    train_loop(cfg, args.tmp_csv, args.amp_csv, args.out_dir)