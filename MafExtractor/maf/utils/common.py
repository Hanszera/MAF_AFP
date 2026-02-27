from __future__ import annotations
import os, json, math, random
import torch




def set_seed(seed: int = 42):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def load_yaml_text(text: str) -> dict:
    import yaml
    return yaml.safe_load(text)




def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)