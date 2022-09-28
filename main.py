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
from sklearn.preprocessing import MinMaxScaler

from dataset import StockDatasetSW_multistep, StockDatasetSW_singlestep, YahooDatasetSW_singlestep
from model import Transformer, TransformerDecoder, TransformerDecoder_v2, WeatherLSTM
from eval_plot import eval_mae, eval_mae_decoder, plot_scores
from utils import scaler, MyMinMaxScaler


sp500_dataset_path = "datasets/spx.csv"
yahoo_dataset_path = "datasets/yahoo_stock.csv"
sp500 = pd.read_csv(sp500_dataset_path)
yahoo = pd.read_csv(yahoo_dataset_path)

sp500_data = sp500['close'].to_numpy()
sp500_data = torch.from_numpy(sp500_data).to(torch.float32)
sp500_trainset = sp500_data[0:int(len(sp500_data) * 0.7)]
sp500_testset = sp500_data[int(len(sp500_data) * 0.7):]
sp500_trainset_scaled, sp500_testset_scaled = scaler(sp500_trainset, sp500_testset)

date_time = pd.to_datetime(yahoo['Date'], format='%Y.%m.%d')
day = 24*60*60
year = (365.2425)*day
timestamp_s = date_time.map(pd.Timestamp.timestamp)
yahoo['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
yahoo['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
yahoo['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
yahoo['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

yahoo_data = yahoo[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'Day sin', 'Day cos', 'Year sin', 'Year cos']]
yahoo_class_idx = 3
yahoo_trainset = yahoo_data.iloc[:int(len(yahoo_data) * 0.7)]
yahoo_testset = yahoo_data.iloc[int(len(yahoo_data) * 0.7):]
scaler = MinMaxScaler()
scaler.fit(yahoo_trainset)
yahoo_trainset_scaled = scaler.transform(yahoo_trainset)
yahoo_testset_scaled = scaler.transform(yahoo_testset)

model_type = "decoder_v2"

if model_type == "transformer":

    trainset, testset = sp500_trainset_scaled, sp500_testset_scaled

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    learning_rate = 0.01
    epochs = 10
    window_len = 7
    output_len = 3
    train_dataset = StockDatasetSW_multistep(trainset, window_len, output_len)
    test_dataset = StockDatasetSW_multistep(testset, window_len, output_len)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = Transformer(seq_len=window_len, num_encoder=6, num_decoder=6, input_size=1, output_size=output_len, d_model=512, num_heads=8, feedforward_dim=1024).to(device)
    loss_fun = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_maes = []
    test_maes = []
    for e in tqdm(range(epochs)):
        model.eval()
        train_mae = eval_mae(model, train_dl, device)
        test_mae = eval_mae(model, test_dl, device)
        train_maes.append(train_mae.cpu())
        test_maes.append(test_mae.cpu())
        print(f"Epoch {e} - Train MAE {train_mae} - Test MAE {test_mae}")
        model.train()
        for i, (src, trg, trg_y) in enumerate(train_dl):
            src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
            optimizer.zero_grad()
            out = model(src, trg)
            loss = loss_fun(out, trg_y)
            if i % 50 == 0:
                print(f'loss {loss.cpu().item():.3f}')
            loss.backward()
            optimizer.step()
    plot_scores(train_maes, test_maes)

elif model_type == "decoder":

    trainset, testset = yahoo_trainset.to_numpy(), yahoo_testset.to_numpy()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    learning_rate = 0.01
    epochs = 50
    window_len = 7
    input_size = 10
    output_size = 1
    d_model = 20
    train_dataset = YahooDatasetSW_singlestep(trainset, window_len, yahoo_class_idx)
    test_dataset = YahooDatasetSW_singlestep(testset, window_len, yahoo_class_idx)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = TransformerDecoder(seq_len=window_len, num_layer=1, input_size=input_size, output_size=output_size, d_model=d_model, num_heads=1, feedforward_dim=32).to(device)
    loss_fun = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_maes = []
    test_maes = []
    losses = []
    for e in tqdm(range(epochs)):
        model.eval()
        train_mae = eval_mae_decoder(model, train_dl, device)
        test_mae = eval_mae_decoder(model, test_dl, device)
        train_maes.append(train_mae.cpu())
        test_maes.append(test_mae.cpu())
        print(f"Epoch {e} - Train MAE {train_mae} - Test MAE {test_mae}")
        model.train()
        avg_loss = 0
        count = 0
        for i, (src, trg) in enumerate(train_dl):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src)
            loss = loss_fun(out, trg)
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 50 == 0:
                print(f'loss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        losses.append(avg_loss)
    plot_scores(train_maes, test_maes, losses)

elif model_type == "decoder_v2":
    trainset, testset = yahoo_trainset.to_numpy(), yahoo_testset.to_numpy()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    learning_rate = 0.01
    epochs = 50
    window_len = 7
    input_size = 10
    output_size = 1
    train_dataset = YahooDatasetSW_singlestep(trainset, window_len, yahoo_class_idx)
    test_dataset = YahooDatasetSW_singlestep(testset, window_len, yahoo_class_idx)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = TransformerDecoder_v2(seq_len=window_len, num_layer=1, input_size=input_size, output_size=output_size, num_heads=1, feedforward_dim=32).to(device)
    loss_fun = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_maes = []
    test_maes = []
    losses = []
    for e in tqdm(range(epochs)):
        model.eval()
        train_mae = eval_mae_decoder(model, train_dl, device)
        test_mae = eval_mae_decoder(model, test_dl, device)
        train_maes.append(train_mae.cpu())
        test_maes.append(test_mae.cpu())
        print(f"Epoch {e} - Train MAE {train_mae} - Test MAE {test_mae}")
        model.train()
        avg_loss = 0
        count = 0
        for i, (src, trg) in enumerate(train_dl):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src)
            loss = loss_fun(out, trg)
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 10 == 0:
                print(f'loss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        losses.append(avg_loss)
    plot_scores(train_maes, test_maes, losses)

elif model_type == "lstm":
    trainset, testset = yahoo_trainset.to_numpy(), yahoo_testset.to_numpy()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    learning_rate = 0.01
    epochs = 50
    window_len = 7
    input_size = 10
    output_size = 1
    d_model = 20
    train_dataset = YahooDatasetSW_singlestep(trainset, window_len, yahoo_class_idx)
    test_dataset = YahooDatasetSW_singlestep(testset, window_len, yahoo_class_idx)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = WeatherLSTM(input_size, d_model, output_size).to(device)

    loss_fun = nn.L1Loss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    train_maes = []
    test_maes = []
    losses = []
    for e in tqdm(range(epochs)):

        model.eval()

        train_mae = eval_mae_decoder(model, train_dl, device)
        test_mae = eval_mae_decoder(model, test_dl, device)
        train_maes.append(train_mae.cpu())
        test_maes.append(test_mae.cpu())

        print(f'Epoch {e:03d} - Train MAE {train_mae:.3f} - Test MAE {test_mae:.3f}')

        avg_loss = 0
        count = 0
        model.train()
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            y_pred = model(x)
            loss = loss_fun(y_pred, y)
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 10 == 0:
                print(f'loss {loss.cpu().item():.3f}')
            loss.backward()
            opt.step()
            count += 1
        avg_loss /= count
        losses.append(avg_loss)
    plot_scores(train_maes, test_maes, losses)
