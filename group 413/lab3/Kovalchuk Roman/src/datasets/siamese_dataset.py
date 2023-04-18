import random

import torch
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, x_t = self.dataset[index]
        is_diff = random.randint(0, 1)

        while True:
            idx2 = random.randint(0, len(self) - 1)
            y, y_t = self.dataset[idx2]
            if is_diff and x_t != y_t:
                break
            if not is_diff and x_t == y_t:
                break

        return x, y, torch.Tensor([x_t == y_t])

    def __len__(self):
        return len(self.dataset)
