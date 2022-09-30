import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SP500Dataset, YahooDataset, YahooDataset2, YahooDatasetStd
from model import Transformer, TransformerDecoder, TransformerDecoder_v2, WeatherLSTM
from eval import eval_mae, eval_mae2, eval_mae_std
from plot import plot_scores
from train import train_model, train_model2, train_model_std
from test import test, test2, test_std

sp500_dataset_path = "datasets/spx.csv"
yahoo_dataset_path = "datasets/yahoo_stock.csv"
predictions_path = "predictions/"
training_results_path = "training_results/"

model_type = "transformer"

if model_type == "transformer":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 1
    window_len = 365
    forecast_len = 60
    input_size = 5
    num_layer = 1
    output_size = 1
    d_model = 32
    dropout = 0
    
    train_dataset = YahooDatasetStd(yahoo_dataset_path, window_len, forecast_len, train=True)
    scaler = train_dataset.get_scaler()
    test_dataset = YahooDatasetStd(yahoo_dataset_path, window_len, forecast_len, train=False, scaler=scaler)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    model = Transformer(seq_len=window_len, num_encoder=num_layer, num_decoder=num_layer, input_size=input_size, output_size=output_size, d_model=d_model, num_heads=d_model, feedforward_dim=64, dropout=dropout).to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model, results = train_model_std(device, model, train_dl, test_dl, num_epochs, loss_fn, eval_mae_std, optimizer)
    
    plot_scores(results['train_scores'], results['test_scores'], results['losses'], training_results_path + model_type + "/")
    test_std(device, model, test_dl, forecast_len, scaler, save_path=predictions_path + model_type + "/")

elif model_type == "decoder":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 1
    window_len = 365
    forecast_len = 60
    input_size = 5
    num_layer = 1
    output_size = 1
    d_model = 32
    dropout = 0
    
    train_dataset = YahooDataset2(yahoo_dataset_path, window_len, forecast_len, train=True)
    scaler = train_dataset.get_scaler()
    test_dataset = YahooDataset2(yahoo_dataset_path, window_len, forecast_len, train=False, scaler=scaler)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    model = TransformerDecoder(seq_len=window_len, num_layer=num_layer, input_size=input_size, output_size=output_size, d_model=d_model, num_heads=d_model, feedforward_dim=64, dropout=dropout).to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model, results = train_model2(device, model, train_dl, test_dl, num_epochs, loss_fn, eval_mae2, optimizer)
    
    plot_scores(results['train_scores'], results['test_scores'], results['losses'], training_results_path + model_type + "/")
    test2(device, model, test_dl, forecast_len, scaler, save_path=predictions_path + model_type + "/")

elif model_type == "decoder_v2":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    learning_rate = 0.001
    num_epochs = 10
    window_len = 360
    forecast_len = 60
    input_size = 5
    num_layer = 1
    output_size = 1
    dropout = 0
    
    train_dataset = YahooDataset2(yahoo_dataset_path, window_len, forecast_len, train=True)
    scaler = train_dataset.get_scaler()
    test_dataset = YahooDataset2(yahoo_dataset_path, window_len, forecast_len, train=False, scaler=scaler)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    model = TransformerDecoder_v2(seq_len=window_len, num_layer=num_layer, input_size=input_size, output_size=output_size, num_heads=input_size, feedforward_dim=2048, dropout=dropout).to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model, results = train_model2(device, model, train_dl, test_dl, num_epochs, loss_fn, eval_mae2, optimizer)
    
    plot_scores(results['train_scores'], results['test_scores'], results['losses'], training_results_path + model_type + "/")
    test2(device, model, test_dl, forecast_len, scaler, save_path=predictions_path + model_type + "/")
    

elif model_type == "lstm":
    
    # TODO
    pass
