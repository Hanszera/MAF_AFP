from torch.utils.data import Dataset
from MafExtractor.maf.utils.aa import seq_to_tensor
import torch
class XYDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.size = len(dataFrame)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'sequences': self.dataFrame['sequence'].iloc[idx],
                'labels': self.dataFrame['label'].iloc[idx]}

class XYDataset_MAF(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.size = len(dataFrame)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        seq = self.dataFrame['sequence'].iloc[idx]
        maf_X = seq_to_tensor(seq)

        return {'sequences': self.dataFrame['sequence'].iloc[idx],
                'labels': self.dataFrame['label'].iloc[idx],
                'maf_X': maf_X}


class XDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.size = len(dataFrame)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {'sequences': self.dataFrame['sequence'].iloc[idx]}
