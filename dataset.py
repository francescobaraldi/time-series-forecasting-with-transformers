import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset


class SP500Dataset(Dataset):
    def prepare_dataset(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        dataset = dataset[['close']]
        return dataset
    
    def split_dataset(self, dataset, train, train_rate):
        n = len(dataset)
        if train:
            return dataset[:int(n * train_rate)]
        else:
            return dataset[int(n * (1 - train_rate)):]
    
    def __init__(self, dataset_path, window_len, train=True, train_rate=0.7, scaler=None):
        super().__init__()
        dataset = self.prepare_dataset(dataset_path)
        self.dataset = self.split_dataset(dataset, train, train_rate).to_numpy()
        self.window_len = window_len
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(dataset.to_numpy())
        self.scaler = scaler
        
    def __len__(self):
        return len(self.dataset) - self.window_len - 1
    
    def get_scaler(self):
        return self.scaler
    
    def __getitem__(self, index):
        src = self.dataset[index:index + self.window_len]
        index += 1
        trg = self.dataset[index:index + self.window_len]
        src = self.scaler.transform(src.reshape(-1, 1))
        trg = self.scaler.transform(trg.reshape(-1, 1))
        return torch.from_numpy(src).unsqueeze(-1), torch.from_numpy(trg).unsqueeze(-1)


class YahooDataset(Dataset):
    def prepare_dataset(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        
        date_time = pd.to_datetime(dataset['Date'], format='%Y.%m.%d')
        day = 24*60*60
        year = (365.2425)*day
        timestamp_s = date_time.map(pd.Timestamp.timestamp)
        dataset['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        dataset['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        dataset['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        dataset['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

        # dataset = dataset[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'Day sin', 'Day cos', 'Year sin', 'Year cos']]
        dataset = dataset[['Close', 'Day sin', 'Day cos', 'Year sin', 'Year cos']]
        class_idx = 0
        return dataset, class_idx
    
    def split_dataset(self, dataset, train, train_rate):
        n = len(dataset)
        if train:
            return dataset[:int(n * train_rate)]
        else:
            return dataset[int(n * (1 - train_rate)):]
    
    def __init__(self, dataset_path, window_len, forecast_len=1, train=True, train_rate=0.7, scaler=None):
        super().__init__()
        dataset, class_idx = self.prepare_dataset(dataset_path)
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
        src = self.dataset[index:index + self.window_len, :]
        trg_forecast = self.dataset[index + self.window_len: index + self.window_len + self.forecast_len, :]
        index += self.forecast_len
        trg = self.dataset[index:index + self.window_len, :]
        src = self.scaler.transform(src)
        trg = self.scaler.transform(trg)
        trg_forecast = self.scaler.transform(trg_forecast)
        return torch.from_numpy(src).to(torch.float32), torch.from_numpy(trg).to(torch.float32), self.class_idx


class YahooDatasetInference(Dataset):
    def prepare_dataset(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        
        date_time = pd.to_datetime(dataset['Date'], format='%Y.%m.%d')
        day = 24*60*60
        year = (365.2425)*day
        timestamp_s = date_time.map(pd.Timestamp.timestamp)
        dataset['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        dataset['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        dataset['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        dataset['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

        # dataset = dataset[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'Day sin', 'Day cos', 'Year sin', 'Year cos']]
        dataset = dataset[['Close', 'Day sin', 'Day cos', 'Year sin', 'Year cos']]
        class_idx = 0
        return dataset, class_idx
    
    def split_dataset(self, dataset, test_rate):
        n = len(dataset)
        return dataset[:int(n * test_rate)]
    
    def __init__(self, dataset_path, window_len, scaler, forecast_len=1, test_rate=0.3):
        super().__init__()
        dataset, class_idx = self.prepare_dataset(dataset_path)
        self.class_idx = class_idx
        self.forecast_len = forecast_len
        self.dataset = self.split_dataset(dataset, test_rate).to_numpy()
        self.window_len = window_len
        self.scaler = scaler
        
    def __len__(self):
        return len(self.dataset) - self.window_len - self.forecast_len
    
    def get_scaler(self):
        return self.scaler
    
    def __getitem__(self, index):
        src = self.dataset[index:index + self.window_len, :]
        trg = self.dataset[index + self.window_len: index + self.window_len + self.forecast_len, :]
        src = self.scaler.transform(src)
        trg = self.scaler.transform(trg)
        return torch.from_numpy(src).to(torch.float32), torch.from_numpy(trg).to(torch.float32), self.class_idx

