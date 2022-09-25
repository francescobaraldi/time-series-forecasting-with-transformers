import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from tqdm import tqdm

from dataset import StockDatasetSW
from model import Transformer, eval, DotProductAttention

# d = 1
# model = DotProductAttention()
# queries = torch.rand((32, 7, d))
# keys = torch.rand((32, 7, d))
# values = torch.rand((32, 7, 1024))
# out = model(queries, keys, values)


sp500 = pd.read_csv("datasets/spx.csv")
sp500.head()
# plt.plot(sp500['close'])
# plt.show()

# dates = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in sp500['Date']]
# dates = mdates.drange(dates[0], dates[-1], datetime.timedelta(days=30))
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
# plt.plot(dates, sp500['Close'])
# plt.gcf().autofmt_xdate()
# plt.show()

data = sp500['close'].to_numpy()
data = torch.from_numpy(data).to(torch.float32)

batch_size = 32
learning_rate = 0.01
epochs = 10
window_len = 7
output_len = 1
trainset = data[0:int(len(data) * 0.7)]
testset = data[int(len(data) * 0.7):]
train_dataset = StockDatasetSW(trainset, window_len, output_len)
test_dataset = StockDatasetSW(testset, window_len, output_len)
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
model = Transformer(seq_len=window_len, num_encoder=6, input_size=1, embed_dim=512, num_heads=1, feedforward_dim=1024)
loss_fun = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for e in range(epochs):
    model.eval()
    
    train_mae = eval(model, train_dl)
    test_mae = eval(model, test_dl)
    
    print(f"Epoch {e} - Train MAE {train_mae} - Test MAE {test_mae}")
    
    model.train()
    for seq, trg in tqdm(train_dl):
        optimizer.zero_grad()
        seq_mask = torch.ones_like(seq)
        out = model(seq, seq_mask)
        loss = loss_fun(out, trg)
        
        loss.backward()
        optimizer.step()

