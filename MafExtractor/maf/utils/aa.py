import torch.nn.functional as F

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA2IDX = {a:i for i,a in enumerate(AA_LIST)}


# AA index：
# [hydropathy, charge(+1/0/-1), helix_propensity, mass_norm, aromatic(0/1), polar(0/1), size_norm, hbond_donor(0/1)]
AA_PROP = {
# Kyte-Doolittle hydropathy (approx), charge at pH ~7
'A': [ 1.8, 0, 1.45, 0.31, 0, 0, 0.31, 0],
'C': [ 2.5, 0, 0.77, 0.55, 0, 0, 0.55, 0],
'D': [-3.5, -1, 0.98, 0.41, 0, 1, 0.41, 0],
'E': [-3.5, -1, 1.53, 0.54, 0, 1, 0.54, 0],
'F': [ 2.8, 0, 1.12, 0.79, 1, 0, 0.79, 0],
'G': [-0.4, 0, 0.53, 0.00, 0, 0, 0.00, 0],
'H': [-3.2, +1, 1.24, 0.62, 0, 1, 0.62, 1],
'I': [ 4.5, 0, 1.00, 0.65, 0, 0, 0.65, 0],
'K': [-3.9, +1, 1.07, 0.59, 0, 1, 0.59, 1],
'L': [ 3.8, 0, 1.34, 0.64, 0, 0, 0.64, 0],
'M': [ 1.9, 0, 1.20, 0.67, 0, 0, 0.67, 0],
'N': [-3.5, 0, 0.73, 0.47, 0, 1, 0.47, 1],
'P': [-1.6, 0, 0.59, 0.49, 0, 0, 0.49, 0],
'Q': [-3.5, 0, 1.17, 0.58, 0, 1, 0.58, 1],
'R': [-4.5, +1, 0.79, 0.70, 0, 1, 0.70, 1],
'S': [-0.8, 0, 0.79, 0.32, 0, 1, 0.32, 1],
'T': [-0.7, 0, 0.82, 0.44, 0, 1, 0.44, 1],
'V': [ 4.2, 0, 1.06, 0.55, 0, 0, 0.55, 0],
'W': [-0.9, 0, 1.14, 1.00, 1, 0, 1.00, 1],
'Y': [-1.3, 0, 0.61, 0.84, 1, 1, 0.84, 1],
}


import torch
def center_align(local_feat, seq_len, kernel_size):
    B, L_conv, C = local_feat.shape
    offset = kernel_size // 2
    aligned = torch.zeros(B, seq_len, C, device=local_feat.device)
    aligned[:, offset:offset + L_conv, :] = local_feat
    return aligned

def seq_to_tensor(seq: str, aa_dim: int = 8) -> torch.Tensor:
    xs = []
    for ch in seq:
        xs.append(AA_PROP.get(ch, AA_PROP['G']))
    x = torch.tensor(xs, dtype=torch.float32)
    if aa_dim != len(x[0]):

        W = torch.nn.functional.normalize(torch.randn(len(AA_PROP['A']), aa_dim), dim=0)
        x = x @ W
    return x # [L, aa_dim]

def onehot_from_seqs(seqs: list[str]) -> torch.Tensor:
    Lmax = max(len(s) for s in seqs)
    if Lmax < 10:
        Lmax = 10


    B = len(seqs)
    idx = torch.full((B, Lmax), fill_value=0, dtype=torch.long)
    for b, s in enumerate(seqs):
        arr = [AA2IDX.get(ch, 0) for ch in s]
        if len(arr) > 0:
            idx[b, :len(arr)] = torch.tensor(arr, dtype=torch.long)
    one = F.one_hot(idx, num_classes=20) # [B,L,20]
    return one.permute(0, 2, 1).contiguous()

def onehot_from_seqs_token(seqs: list[str]) -> torch.Tensor:
    Lmax = max(len(s) for s in seqs)
    if Lmax < 10:
        Lmax = 10
    B = len(seqs)
    idx = torch.full((B, Lmax), fill_value=0, dtype=torch.long)
    for b, s in enumerate(seqs):
        arr = [AA2IDX.get(ch, 0) for ch in s]
        if len(arr) > 0:
            idx[b, :len(arr)] = torch.tensor(arr, dtype=torch.long)
    one = F.one_hot(idx, num_classes=20) # [B,L,20]
    return one.permute(0, 2, 1).contiguous()