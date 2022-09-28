import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SP500Dataset, YahooDataset
from model import Transformer, TransformerDecoder, TransformerDecoder_v2, WeatherLSTM
from eval import eval_mae
from plot import plot_scores, plot_predictions
from train import train_model

sp500_dataset_path = "datasets/spx.csv"
yahoo_dataset_path = "datasets/yahoo_stock.csv"
predictions_path = "predictions/"

model_type = "decoder"

if model_type == "transformer":

    # TODO
    pass

elif model_type == "decoder":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 10
    window_len = 7
    input_size = 10
    output_size = 1
    d_model = 20
    
    train_dataset = YahooDataset(yahoo_dataset_path, window_len, train=True)
    scaler = train_dataset.get_scaler()
    test_dataset = YahooDataset(yahoo_dataset_path, window_len, train=False, scaler=scaler)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    model = TransformerDecoder(seq_len=window_len, num_layer=1, input_size=input_size, output_size=output_size, d_model=d_model, num_heads=1, feedforward_dim=32).to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model, results = train_model(device, model, train_dl, test_dl, num_epochs, loss_fn, eval_mae, optimizer)
    
    plot_scores(results['train_scores'], results['test_scores'], results['losses'])
    plot_predictions(device, model, test_dl, scaler, predictions_path)

elif model_type == "decoder_v2":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 10
    window_len = 7
    input_size = 10
    output_size = 1
    d_model = 20
    
    train_dataset = YahooDataset(yahoo_dataset_path, window_len, train=True)
    scaler = train_dataset.get_scaler()
    test_dataset = YahooDataset(yahoo_dataset_path, window_len, train=False, scaler=scaler)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    model = TransformerDecoder_v2(seq_len=window_len, num_layer=1, input_size=input_size, output_size=output_size, num_heads=1, feedforward_dim=32).to(device)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model, results = train_model(device, model, train_dl, test_dl, num_epochs, loss_fn, eval_mae, optimizer)
    
    plot_scores(results['train_scores'], results['test_scores'], results['losses'])
    plot_predictions(device, model, test_dl, scaler, predictions_path)
    

elif model_type == "lstm":
    
    # TODO
    pass
