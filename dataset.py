import torch
from torch.utils.data import Dataset


class StockDatasetSW_multistep(Dataset):
    def __init__(self, data, window_len, output_len):
        self.data = data
        self.window_len = window_len
        self.output_len = output_len
        
    def __len__(self):
        return len(self.data) - (self.window_len + self.output_len) + 1
    
    def __getitem__(self, index):
        src = self.data[index:index + self.window_len]
        index += self.window_len - 1
        trg = self.data[index:index + self.output_len]
        index += 1
        trg_y = self.data[index:index + self.output_len]
        return src.unsqueeze(-1), trg.unsqueeze(-1), trg_y.unsqueeze(-1)


class StockDatasetSW_singlestep(Dataset):
    def __init__(self, data, window_len):
        self.data = data
        self.window_len = window_len
        
    def __len__(self):
        return len(self.data) - (self.window_len + 1)
    
    def __getitem__(self, index):
        src = self.data[index:index + self.window_len]
        index += 1
        trg = self.data[index:index + self.window_len]
        return src.unsqueeze(-1), trg.unsqueeze(-1)


class YahooDatasetSW_singlestep(Dataset):
    def __init__(self, data, window_len, class_idx):
        self.data = data
        self.window_len = window_len
        self.class_idx = class_idx
        
    def __len__(self):
        return len(self.data) - (self.window_len + 1)
    
    def __getitem__(self, index):
        src = self.data[index:index + self.window_len, :]
        index += 1
        trg = self.data[index:index + self.window_len, self.class_idx]
        return torch.from_numpy(src).to(torch.float32), torch.from_numpy(trg).unsqueeze(-1).to(torch.float32)
