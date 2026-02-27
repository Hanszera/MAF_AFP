from dataclasses import dataclass
from typing import Optional, List, Tuple
import csv, math, random
import torch
from torch.utils.data import Dataset, DataLoader
from maf.utils.aa import seq_to_tensor
import os
import pandas as pd




@dataclass
class Sample:
    seq: str
    label_windows: List[torch.Tensor] # 每尺度 [L_w]，1=插膜窗口；0=非插膜；-1=未知




def parse_tm_regions(s: str) -> List[Tuple[int,int]]:
    # "3-22;40-60" → [(3,22),(40,60)] 1-based inclusive
    out = []
    if not s:
        return out
    for it in s.split(';'):
        if '-' in it:
            a,b = it.split('-')
            out.append((int(a), int(b)))
    return out

# def weak_labels_by_threshold(seq: str, window_sizes: List[int], thr_pos: float, thr_neg: float) -> List[torch.Tensor]:
#     """用简化的“平均疏水性”替代 ΔG_if 产生弱标签。
#     实际可替换为 Wimley–White 尺度。
#     返回每尺度的标签向量（1/0/-1）。
#     """
#     import numpy as np
#     from maf.utils.aa import AA_PROP
#     hydro = np.array([AA_PROP.get(ch, AA_PROP['G'])[0] for ch in seq], dtype=float)
#     labels = []
#     for w in window_sizes:
#         Lw = max(0, len(seq)-w+1)
#         arr = np.convolve(hydro, np.ones(w)/w, mode='valid') if Lw>0 else np.array([])
#         y = np.full((Lw,), -1, dtype=int)
#         if Lw>0:
#             y[arr<=thr_pos] = 1
#             y[arr>=thr_neg] = 0
#         labels.append(torch.tensor(y, dtype=torch.long))
#     return labels


def weak_labels_by_threshold(seq: str, window_sizes: List[int],
                             thr_pos: float, thr_neg: float) -> List[torch.Tensor]:
    """
    使用 Wimley–White ΔG_if（界面插入自由能）产生弱标签。
    ΔG_if 越负 → 越有可能是 AFP/AMP 的疏水插膜段。

    返回每尺度标签：
        1  → 可能是 AFP-like 插膜片段（ΔG_if 很负）
        0  → 可能是非 AFP 片段（ΔG_if 很正）
       -1  → 不确定，忽略
    """
    import numpy as np
    from maf.utils.biophys import WW_IF

    # 1) 映射到 ΔG_if 数组
    dgif = np.array([WW_IF.get(aa, 0.2) for aa in seq], dtype=float)

    labels = []
    for w in window_sizes:
        Lw = len(seq) - w + 1
        if Lw <= 0:
            labels.append(torch.tensor([], dtype=torch.long))
            continue

        # 2) 滑动窗口平均 ΔG_if
        arr = np.convolve(dgif, np.ones(w) / w, mode='valid')

        # 3) 初始化
        y = np.full((Lw,), -1, dtype=int)

        # 4) 物理意义阈值判断：
        #    arr 越负越可能是插膜区 → AFP-like
        y[arr <= thr_pos] = 1  # ΔG_if 非常负 → 插膜段
        y[arr >= thr_neg] = 0  # ΔG_if 正或偏大 → 不插膜

        labels.append(torch.tensor(y, dtype=torch.long))

    return labels


def labels_from_tm_regions(seq: str, tm_regions: str, window_sizes: List[int]) -> List[torch.Tensor]:
    import ast
    L = len(seq)
    labels = []
    for w in window_sizes:
        Lw = L - w + 1
        y = torch.full((max(0,Lw),), -1, dtype=torch.long)
        for (a,b) in ast.literal_eval(tm_regions):
            # 将覆盖窗口标为 1（插膜）
            a0,b0 = a-1,b-1
            for i in range(0, max(0,Lw)):
                if not (i > b0 or i+w-1 < a0):
                    y[i] = 1
        # 未标注的窗口默认 0（非插膜）
        y[y==-1] = 0
        labels.append(y)
    return labels

from torch.utils.data import Sampler


class BucketingSampler(Sampler):
    def __init__(self, lengths, batch_size,shuffle=False, bucket_size=200):
        """
        lengths: List[int] 每条序列长度
        bucket_size: 桶宽，例如 200 表示 0–200, 200–400, ...
        """
        self.batch_size = batch_size
        # 建桶：{bucket_id: [indices]}
        buckets = {}
        for idx, L in enumerate(lengths):
            bid = L // bucket_size
            buckets.setdefault(bid, []).append(idx)
        # 每个桶内部 shuffle，并切分为 batch
        self.batches = []
        for bid, inds in buckets.items():
            if shuffle:
                random.shuffle(inds)
            for i in range(0, len(inds), batch_size):
                self.batches.append(inds[i:i+batch_size])
        # 打乱桶之间的 batch 顺序
        if shuffle:
            random.shuffle(self.batches)


    def __len__(self):
        return len(self.batches)


    def __iter__(self):
        for batch in self.batches:
            yield batch

def load_amp_labels(path):
    df = pd.read_csv(path)
    amp_dict = {row['sequence']: int(row['label']) for _, row in df.iterrows()}
    return amp_dict
class MafWindowDataset(Dataset):
    def __init__(self, tmp_csv: Optional[str], amp_csv: Optional[str], window_sizes: List[int],
    weak_pos_thr: float=-1.0, weak_neg_thr: float=0.5, max_len: int=256,
    split: str='train', split_ratio: float=0.85, seed: int=42):
        super().__init__()
        random.seed(seed)
        self.samples: List[Sample] = []
        self.amp_labels_dict = load_amp_labels(amp_csv)
        # 读 TMP 强监督
        tmp_rows = []
        if tmp_csv and os.path.exists(tmp_csv):
            import pandas as pd
            df = pd.read_csv(tmp_csv)
            tmp_rows = df.to_dict('records')
        # 读 AMP 弱监督
        amp_rows = []
        if amp_csv and os.path.exists(amp_csv):
            import pandas as pd
            df = pd.read_csv(amp_csv)
            amp_rows = df.to_dict('records')
    # 切分
        def split_rows(rows):
            k = int(len(rows)*split_ratio)
            return rows[:k], rows[k:]
        tmp_tr, tmp_te = split_rows(tmp_rows)
        amp_tr, amp_te = split_rows(amp_rows)
        rows = (tmp_tr + amp_tr) if split=='train' else (tmp_te + amp_te)
        for r in rows:
            seq = str(r['sequence'])
            if 'tm_regions' in r and isinstance(r['tm_regions'], str) and len(r['tm_regions'])>0:
                labs = labels_from_tm_regions(seq, r['tm_regions'], window_sizes)
            else:
                labs = weak_labels_by_threshold(seq, window_sizes, weak_pos_thr, weak_neg_thr)
            self.samples.append(Sample(seq=seq, label_windows=labs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = seq_to_tensor(s.seq) # [L, D]
        amp_labels = self.amp_labels_dict.get(self.samples[idx].seq, 0)
        return s.seq, x, s.label_windows,amp_labels



def collate_fn(batch):
    seqs, xs, labs,amp_labels = zip(*batch)
    maxL = max(x.size(0) for x in xs)
    D = xs[0].size(1)
    X = torch.zeros(len(xs), maxL, D)
    for i,x in enumerate(xs):
        X[i,:x.size(0)] = x
    return seqs, X, labs,amp_labels