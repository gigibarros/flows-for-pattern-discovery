import random
import torch
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class FluorData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # convert to float tensor
        self.y = torch.tensor(y, dtype=torch.long)     # class labels as long tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

def get_fluor_dataset(X, y):
    return FluorData(X, y)