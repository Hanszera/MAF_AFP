import math
from typing import List
import torch.nn.functional as F
import torch
import torch.nn as nn
from MafExtractor.maf.utils.aa import AA_LIST


# Wimley–White interfacial scale（kcal/mol，示例值，建议按需替换为你校准的表）
WW_IF = {
'A': 0.17, 'C': -0.24, 'D': 3.49, 'E': 2.68, 'F': -1.13,
'G': 0.01, 'H': 2.33, 'I': -0.31, 'K': 2.80, 'L': -0.56,
'M': -0.10,'N': 2.05, 'P': 0.45, 'Q': 2.36, 'R': 2.58,
'S': 0.13, 'T': 0.14, 'V': 0.07, 'W': -1.85,'Y': -0.94,
}


# Eisenberg hydrophobicity
EISENBERG = {
'A': 0.62, 'C': 0.29, 'D': -0.90,'E': -0.74,'F': 1.19,
'G': 0.48, 'H': -0.40,'I': 1.38, 'K': -1.50,'L': 1.06,
'M': 0.64, 'N': -0.78,'P': 0.12, 'Q': -0.85,'R': -2.53,
'S': -0.18,'T': -0.05,'V': 1.08, 'W': 0.81,'Y': 0.26,
}


ALPHA_HELIX_DEG_STEP = 100.0




def ww_dG_if_window(seq: str, w: int) -> List[float]:

    L = len(seq)
    if L < w:
        return []
    vals = [WW_IF.get(ch, 0.5) for ch in seq]
    s = sum(vals[:w])
    out = [s]
    for i in range(w, L):
        s += vals[i] - vals[i-w]
        out.append(s)

    return [v/float(w) for v in out]




def hydrophobic_moment_window(seq: str, w: int, scale: str='eisenberg') -> List[float]:

    L = len(seq)
    if L < w:
        return []
    table = EISENBERG if scale=='eisenberg' else EISENBERG
    hs = [table.get(ch, 0.0) for ch in seq]
    out = []
    rad = math.pi/180.0
    for start in range(0, L-w+1):
        mx, my = 0.0, 0.0
        for i in range(w):
            angle = (i * ALPHA_HELIX_DEG_STEP) * rad
            mx += hs[start+i] * math.cos(angle)
            my += hs[start+i] * math.sin(angle)
        mu = math.sqrt(mx*mx + my*my) / float(w)
        out.append(mu)
    return out



class SlidingWindowConv(nn.Module):

    def __init__(self, aa_dim: int, window_sizes: List[int], learnable_scales: bool = True):
        super().__init__()
        self.window_sizes = window_sizes
        self.kernels = nn.ParameterList()
        for w in window_sizes:
            # [out_channels=1, in_channels=aa_dim, kernel_size=w]
            k = torch.randn(1, aa_dim, w) * 0.01

            with torch.no_grad():
                k[:] = 1.0 / w
            p = nn.Parameter(k, requires_grad=learnable_scales)
            self.kernels.append(p)


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, D, L]
        outs = []
        for k in self.kernels:
            y = F.conv1d(x, k, stride=1, padding=0) # [B, 1, L-w+1]
            outs.append(y)
        return outs

class WWConv(nn.Module):
    def __init__(self, window_sizes):
        super().__init__()
        self.windows = window_sizes

        self.kernels = nn.ModuleList()
        for w in window_sizes:
            # WW scale shape [20]
            ww = torch.tensor([WW_IF[a] for a in AA_LIST], dtype=torch.float32)
            # conv weight: [1,20,w]
            k = torch.zeros(1, 20, w)
            k[0,:, :] = ww[:,None] / w  # mean
            conv = nn.Conv1d(20, 1, w, bias=False)
            conv.weight.data = k
            conv.weight.requires_grad_(False)
            self.kernels.append(conv)

    def forward(self, onehot):
        # onehot: [B,20,L]
        onehot = onehot.float()
        outputs = []
        for conv in self.kernels:
            # [B, 1, L-w+1]
            outputs.append(conv(onehot))
        return outputs

class HMConv(nn.Module):

    def __init__(self, window_sizes):
        super().__init__()
        self.windows = window_sizes
        self.hscale = torch.tensor([EISENBERG[a] for a in AA_LIST], dtype=torch.float32)

        self.conv_cos = nn.ModuleList()
        self.conv_sin = nn.ModuleList()

        for w in window_sizes:
            angles = torch.arange(w) * (100 * math.pi/180)
            cos_k = torch.cos(angles)
            sin_k = torch.sin(angles)

            # conv weight [1,20,w]
            base = self.hscale[:, None]

            kcos = base * cos_k[None,:] / w
            ksin = base * sin_k[None,:] / w

            conv_c = nn.Conv1d(20, 1, w, bias=False)
            conv_s = nn.Conv1d(20, 1, w, bias=False)

            conv_c.weight.data = kcos[None,:,:]
            conv_s.weight.data = ksin[None,:,:]

            conv_c.weight.requires_grad_(False)
            conv_s.weight.requires_grad_(False)

            self.conv_cos.append(conv_c)
            self.conv_sin.append(conv_s)

    def forward(self, onehot):
        onehot = onehot.float()
        outs = []
        for cc, ss in zip(self.conv_cos, self.conv_sin):
            mx = cc(onehot)   # [B,1,Lw]
            my = ss(onehot)
            hm = torch.sqrt(mx*mx + my*my)
            outs.append(hm)
        return outs