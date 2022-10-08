import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset


class YahooDataset(Dataset):
    def prepare_dataset(self, dataset_path, positional_encoding, input_size):
        dataset = pd.read_csv(dataset_path)
        
        if positional_encoding == "none":
            date_time = pd.to_datetime(dataset['Date'], format='%Y.%m.%d')
            day = 24*60*60
            year = (365.2425)*day
            timestamp_s = date_time.map(pd.Timestamp.timestamp)
            dataset['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
            dataset['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
            dataset['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
            dataset['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
            dataset = dataset[['Close', 'Day sin', 'Day cos', 'Year sin', 'Year cos']]
        else:
            if input_size == 1:
                dataset = dataset[['Close']]
            else:
                dataset = dataset[['Close', 'High', 'Low', 'Open', 'Volume']]
        class_idx = 0
        return dataset, class_idx
    
    def split_dataset(self, dataset, train, train_rate):
        n = len(dataset)
        if train:
            return dataset[:int(n * train_rate)]
        else:
            return dataset[int(n * train_rate):]
    
    def __init__(self, dataset_path, window_len, forecast_len=1, input_size=1, positional_encoding="none", train=True, train_rate=0.7, scaler=None):
        super().__init__()
        
        if positional_encoding != "none" and positional_encoding != "sinusoidal" and positional_encoding != "learnable":
            raise Exception("Positional encoding type not recognized: use 'none', 'sinusoidal' or 'learnable'.")
        
        if input_size != 1 and input_size != 5:
            raise Exception("Input size must be either 1 or 5.")
        self.input_size = input_size
        
        dataset, class_idx = self.prepare_dataset(dataset_path, positional_encoding, input_size)
        self.class_idx = class_idx
        self.forecast_len = forecast_len
        self.dataset = self.split_dataset(dataset, train, train_rate).to_numpy()
        self.window_len = window_len
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(self.dataset)
        self.scaler = scaler
        
    def __len__(self):
        return len(self.dataset) - self.window_len - self.forecast_len
    
    def get_scaler(self):
        return self.scaler
    
    def __getitem__(self, index):
        input = self.dataset[index:index + self.window_len + 1, :]
        index += self.forecast_len
        trg = self.dataset[index:index + self.window_len, :]
        input = self.scaler.transform(input)
        trg = self.scaler.transform(trg)
        return torch.from_numpy(input).to(torch.float32), torch.from_numpy(trg).to(torch.float32), self.class_idx
