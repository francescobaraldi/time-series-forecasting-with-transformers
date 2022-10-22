import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset


class YahooDataset(Dataset):
    def prepare_dataset(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        dataset = dataset[['Close']]
        class_idx = 0
        return dataset, class_idx
    
    def split_dataset(self, dataset, train, train_rate):
        n = len(dataset)
        if train:
            return dataset[:int(n * train_rate)]
        else:
            return dataset[int(n * train_rate):]
    
    def __init__(self, dataset_path, window_len, forecast_len=1, train=True, train_rate=0.7, scalertype="none", scaler=None):
        super().__init__()
        
        
        if train == True and (scalertype != "none" and scalertype != "minmax"):
            raise Exception("scaler not recognized, use 'none' or 'minmax'")
        
        dataset, class_idx = self.prepare_dataset(dataset_path)
        self.class_idx = class_idx
        self.forecast_len = forecast_len
        self.dataset = self.split_dataset(dataset, train, train_rate).to_numpy()
        self.window_len = window_len
        self.scaler = scaler
        if train == True and scalertype == "minmax":
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.dataset)
        
    def __len__(self):
        return len(self.dataset) - self.window_len - 2 * self.forecast_len + 1
    
    def get_scaler(self):
        return self.scaler
    
    def __getitem__(self, index):
        input = self.dataset[index:index + self.window_len + 2 * self.forecast_len - 1, :]
        if self.scaler is not None:
            input = self.scaler.transform(input)
        return torch.from_numpy(input).to(torch.float32), self.window_len, self.class_idx
