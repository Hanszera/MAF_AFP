from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from maf.utils.aa import AA_LIST,onehot_from_seqs_token
from maf.utils.biophys import SlidingWindowConv, WWConv, HMConv





class MafExtractor(nn.Module):
    """MAF 提取器：
    - multi-scale conv 近似 ΔG_if/HM（这里简化为线性窗口打分 + 非线性映射）
    - 片段级打分 → pooling 得到全局特征
    - 监督：窗口级二分类 + ΔG_if 回归（可选）
    输出：
    dict(
    local_scores: List[[B, L_w]]，每尺度的窗口插膜分数
    global_feat: [B, H]
    )
    """
    def __init__(self, aa_dim=8, hidden=128, window_sizes=[10,14,18], learnable_scales=True, dropout=0.1):
        super().__init__()
        in_global = len(window_sizes) * 3 # min/mean/topk 聚合
        self.aa_dim = aa_dim
        self.window_sizes = window_sizes
        self.conv = SlidingWindowConv(aa_dim, window_sizes, learnable_scales)
        self.proj = nn.Sequential(
        nn.Linear(in_global, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden)
        )
        # 窗口级分类头（插膜/非插膜）与回归头（ΔG_if proxy）
        self.cls_heads = nn.ModuleList([nn.Linear(1, 1) for _ in window_sizes])
        self.reg_heads = nn.ModuleList([nn.Linear(1, 1) for _ in window_sizes])

        self.use_physics = True
        self.ww_conv = WWConv(self.window_sizes)
        self.hm_conv = HMConv(self.window_sizes)
        self.phys_proj = nn.Sequential(
            nn.Linear(len(self.window_sizes) * 2 + 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)
        )

        self.local_proj = nn.Sequential(
            nn.Linear(len(window_sizes), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )
        # 新增 local-level 物理特征融合模块
        self.local_phys_proj = nn.Sequential(
            nn.Linear(len(self.window_sizes) * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )
        self.local_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )



    def forward_windows(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # x: [B, L, D] → [B, D, L]
        x = x.transpose(1, 2)
        convs = self.conv(x) # list of [B,1,L_w]
        cls_scores, reg_scores = [], []
        for y, cl, rg in zip(convs, self.cls_heads, self.reg_heads):
            y_t = y.transpose(1, 2) # [B, L_w, 1]
            cls = cl(y_t) # [B, L_w, 1]
            rg_ = rg(y_t) # [B, L_w, 1]
            cls_scores.append(cls.squeeze(-1)) # [B, L_w]
            reg_scores.append(rg_.squeeze(-1)) # [B, L_w]
        return cls_scores, reg_scores


    def aggregate_global(self, cls_scores: List[torch.Tensor]) -> torch.Tensor:
        feats = []
        for s in cls_scores:
            # min/mean/topk 聚合（这里 topk=3）
            min_v = s.min(dim=1).values
            mean_v = s.mean(dim=1)
            topk_v = torch.topk(s, k=min(3, s.size(1)), dim=1).values.mean(dim=1)
            feats.append(torch.stack([min_v, mean_v, topk_v], dim=1))
        g = torch.cat(feats, dim=1) # [B, 3*#scales]
        return self.proj(g)

    @torch.no_grad()
    def physics_global(self, seqs: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        onehot = onehot_from_seqs_token(seqs).to(device)  # [B,20,Lmax]
        B, _, Lmax = onehot.shape
        lengths = torch.tensor([len(s) for s in seqs], device=device)

        ww_list = self.ww_conv(onehot)  # list of [B,1,Lw]
        hm_list = self.hm_conv(onehot)  # list of [B,1,Lw]

        per_scale = []
        best_val = torch.full((B,), float('inf'), device=device)
        best_pos = torch.zeros(B, device=device)
        best_wprop = torch.zeros(B, device=device)
        max_w = max(self.window_sizes)
        eps = 1e-6

        for w, wws, hms in zip(self.window_sizes, ww_list, hm_list):
            wws = wws.squeeze(1)  # [B,Lw]
            hms = hms.squeeze(1)
            Lw = wws.size(1)

            valid_end = torch.clamp(lengths - w + 1, min=0)
            ar = torch.arange(Lw, device=device).unsqueeze(0).expand(B, -1)
            mask = ar < valid_end.unsqueeze(1)

            wws_m = wws.masked_fill(~mask, float('inf'))
            hms_m = hms.masked_fill(~mask, float('-inf'))

            ww_min, ww_idx = wws_m.min(dim=1)
            hm_max, _ = hms_m.max(dim=1)
            per_scale.extend([ww_min, hm_max])

            better = ww_min < best_val
            if better.any():
                center = ww_idx.to(torch.float32) + (w - 1) / 2.0
                pos_norm = center / (lengths.to(torch.float32) - 1.0 + eps)
                best_pos = torch.where(better, pos_norm, best_pos)
                best_wprop = torch.where(better, torch.tensor(w / max_w, device=device), best_wprop)
                best_val = torch.minimum(best_val, ww_min)

        feats = torch.stack(per_scale, dim=1)  # [B, 2*S]
        feats = torch.cat([feats, best_pos.unsqueeze(1), best_wprop.unsqueeze(1)], dim=1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)# [B, 2*S+2]
        return self.phys_proj(feats)

    def residue_level_features(self, cls_scores: List[torch.Tensor], lengths: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Center-align 多尺度 cls_scores 到 residue 位置，返回:
          local_feat: [B, L_max, H]
          local_mask: [B, L_max] (bool tensor, True = valid residue)
        Args:
          cls_scores: list of [B, L_w] for each window size
          lengths: tensor [B] of actual sequence lengths (number of residues, excluding pad)
        """
        device = cls_scores[0].device
        B = cls_scores[0].size(0)
        L_max = lengths.max().item()
        n_scales = len(cls_scores)

        # feats 临时空间： [B, L_max, n_scales]
        feats = torch.zeros(B, L_max, n_scales, device=device, dtype=cls_scores[0].dtype)

        # 对每个尺度（窗口）进行中心对齐
        for j, (s, w) in enumerate(zip(cls_scores, self.window_sizes)):
            # s: [B, L_w]
            L_w = s.size(1)
            # centers: 0..L_w-1 shifted by center offset
            centers = torch.arange(L_w, device=device) + (w // 2)  # shape [L_w]
            # 将每个 batch 的有效位置赋值
            # 我们对每个样本分别选择有效索引（因为每个样本长度可能不同）
            for b in range(B):
                L_real = int(lengths[b].item())
                if L_real <= 0:
                    continue
                # 有效 centers 必须 < L_real
                valid_mask = centers < L_real
                if valid_mask.any():
                    idx_pos = centers[valid_mask]  # tensor of positions
                    # s[b, valid_mask] -> 对应值
                    vals = s[b, valid_mask]
                    feats[b, idx_pos, j] = vals

        # local mask: True 表示真实 residue
        local_mask = (torch.arange(L_max, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)).to(torch.bool)  # [B, L_max]

        # 将尺度维映射到 hidden
        local_feat = self.local_proj(feats)  # [B, L_max, H]

        # 将 padding 路径位置归零（以保证 padding 不携带信息）
        local_feat = local_feat * local_mask.unsqueeze(-1)

        return local_feat, local_mask

    def physics_local(self, seqs: List[str]) -> torch.Tensor:
        """
        计算每个残基的物理特征并进行融合。
        """
        device = next(self.parameters()).device
        onehot = onehot_from_seqs_token(seqs).to(device)
        ww_list = self.ww_conv(onehot)  # list of [B,1,Lw]
        hm_list = self.hm_conv(onehot)  # list of [B,1,Lw]
        B = onehot.size(0)
        Lw_max = max([w.size(2) for w in ww_list])
        lengths = torch.tensor([len(s) for s in seqs], device=device)
        L_max = lengths.max().item()

        physics_maps = []
        for ww, hm in zip(ww_list, hm_list):
            ww_up = torch.nn.functional.interpolate(ww, size=L_max, mode="linear", align_corners=True)
            hm_up = torch.nn.functional.interpolate(hm, size=L_max, mode="linear", align_corners=True)
            physics_maps.append(torch.cat([ww_up, hm_up], dim=1))  # [B, L_max, 2]

        physics_maps = torch.cat(physics_maps, dim=1)
        B, _, Lw_max = physics_maps.size()
        aligned_physics_maps = torch.zeros(B, 2 * len(self.window_sizes), L_max, device=device)
        for i in range(B):
            L_real = lengths[i].item()
            if L_real <= 0:
                continue
            valid_mask = torch.arange(Lw_max, device=device) < L_real
            aligned_physics_maps[i, :, :L_real] = physics_maps[i, :, valid_mask]
        physics_maps_reshaped = aligned_physics_maps.view(-1, 2 * len(self.window_sizes))  # [B * L_max, 2 * S]
        projected_phys_maps = self.local_phys_proj(physics_maps_reshaped)  # [B * L_max, H]
        projected_phys_maps = projected_phys_maps.view(B, Lw_max, -1)
        return projected_phys_maps  # [B, L_max, hidden]

    def forward(self, x: torch.Tensor, seqs: List[str] = None) -> Dict[str, torch.Tensor]:
        cls_scores, reg_scores = self.forward_windows(x)
        g_learned = self.aggregate_global(cls_scores)

        if seqs is not None:
            lengths = torch.tensor([len(s) for s in seqs])
            # clip lengths 不超过 x.size(1)
            lengths = torch.clamp(lengths, min=0, max=x.size(1)).to('cuda')
        else:
            lengths = torch.tensor([x.size(1)] * x.size(0)).to('cuda')
        local_feat,local_mask = self.residue_level_features(cls_scores, lengths)
        if self.use_physics and seqs is not None:
            local_phys = self.physics_local(seqs)
            local_feat = self.local_fusion(torch.cat([local_feat, local_phys], dim=-1))
            g_phys = self.physics_global(seqs)
            g = self.fusion(torch.cat([g_learned, g_phys], dim=1))
        else:
            g = g_learned
        return {
            "local_scores": cls_scores,
            "local_feat": local_feat,
            "local_mask": local_mask,
            "global_feat": g,
            "reg_scores": reg_scores
        }

