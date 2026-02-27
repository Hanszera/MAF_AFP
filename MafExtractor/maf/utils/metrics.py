from typing import Dict
import torch




def binary_f1(logits: torch.Tensor, y: torch.Tensor) -> float:
    p = (torch.sigmoid(logits) > 0.5).long()
    tp = ((p==1) & (y==1)).sum().item()
    fp = ((p==1) & (y==0)).sum().item()
    fn = ((p==0) & (y==1)).sum().item()
    precision = tp / (tp+fp+1e-9)
    recall = tp / (tp+fn+1e-9)
    f1 = 2*precision*recall/(precision+recall+1e-9)
    return f1