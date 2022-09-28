import torch


def scaler(traindata, testdata):
    min = torch.min(traindata)
    max = torch.max(traindata)
    traindata = (traindata - min) / (max - min)
    testdata = (testdata - min) / (max - min)
    return traindata, testdata


class MyMinMaxScaler:
    def __init__(self):
        super().__init__()
        
    def fit(self, data):
        self.data = data
        self.min = torch.min(data)
        self.max = torch.max(data)
    
    def transform(self, x):
        x = (x - self.min) / (self.max - self.min)
        return x
    
    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x
