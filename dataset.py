import torch
from torch.utils.data import Dataset


class StockDatasetSW(Dataset):
    def __init__(self, data, window_len, output_len):
        self.data = data
        self.window_len = window_len
        self.output_len = output_len
        
    def __len__(self):
        return len(self.data) - 7
    
    def __getitem__(self, index):
        src = self.data[index:index + self.window_len]
        index += self.output_len
        trg = self.data[index:index + self.window_len]
        return src.unsqueeze(-1), trg.unsqueeze(-1)
