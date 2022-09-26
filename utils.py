import torch


def scaler(traindata, testdata):
    min = torch.min(traindata)
    max = torch.max(traindata)
    traindata = (traindata - min) / (max - min)
    testdata = (testdata - min) / (max - min)
    return traindata, testdata