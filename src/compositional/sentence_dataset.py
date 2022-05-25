from typing import List, Dict

import torch

from torch.utils.data import Dataset

class SentenceDataset(Dataset):

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}, idx

    def __len__(self):
        return len(self.input_ids)


