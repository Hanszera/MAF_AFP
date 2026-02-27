import random
import torch




def mask_non_insertion(x, window_scores, topk=1, mask_prob=0.15):
    """仅对非插膜窗位置做随机 mask（置零），保持最强插膜窗。
    x: [L, D]; window_scores: [num_windows];
    """
    L = x.size(0)
    x = x.clone()
    # 找到 topk 插膜窗的覆盖区间，避免 mask
    top_idx = torch.topk(window_scores, k=min(topk, len(window_scores)), largest=True).indices
    protected = set()
    for idx in top_idx.tolist():
    # 简单假设 stride=1；实际从外部传入窗口大小以更精确
        start = idx
        end = idx + 1 # 占位：训练时可替换为具体 window size
        for i in range(start, min(end, L)):
            protected.add(i)
    for i in range(L):
        if i not in protected and random.random() < mask_prob:
            x[i].zero_()
    return x




def conservative_mutation(seq: str, tolerance: float = 0.2) -> str:
    """保守突变：在保持电荷与疏水性近似的前提下随机替换。
    简化示例：同侧链/电荷类别内替换。
    """
    groups = {
    'pos': list('KRH'),
    'neg': list('DE'),
    'hydro': list('AVILMFYW'),
    'polar': list('STNQ'),
    'special': list('GP'),
    'other': list('C')
    }
    import random
    out = []
    for ch in seq:
        pick = ch
        r = random.random()
        if ch in groups['pos'] and r < 0.2:
            pick = random.choice(groups['pos'])
        elif ch in groups['neg'] and r < 0.2:
            pick = random.choice(groups['neg'])
        elif ch in groups['hydro'] and r < 0.2:
            pick = random.choice(groups['hydro'])
        elif ch in groups['polar'] and r < 0.2:
            pick = random.choice(groups['polar'])
        elif ch in groups['special'] and r < 0.2:
            pick = random.choice(groups['special'])
        out.append(pick)
    return ''.join(out)