import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SP500Dataset, YahooDataset, YahooDataset2, YahooDatasetStd, YahooDatasetPos
from model import Transformer, TransformerDecoder, TransformerDecoder_v2, TransformerDecoderPos
from eval import eval_mae, eval_mae_std, eval_mae2
from plot import plot_scores
from train import train_model, train_model_std, train_and_test_model, train_model2, train_and_test_model2
from test import test_std

sp500_dataset_path = "datasets/spx.csv"
yahoo_dataset_path = "datasets/yahoo_stock.csv"
predictions_path = "predictions/"
training_results_path = "training_results/"

model_type = "decoder"

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
    
    window_len = 365
    forecast_len = 30
    input_size = 5
    output_size = 1
    train_dataset = YahooDataset2(yahoo_dataset_path, window_len, forecast_len, train=True)
    scaler = train_dataset.get_scaler()
    test_dataset = YahooDataset2(yahoo_dataset_path, window_len, forecast_len, train=False, scaler=scaler)
    model_cls = TransformerDecoder
    loss_fn = nn.MSELoss()
    optim_cls = optim.Adam
    train_fn = train_model2
    eval_fn = eval_mae2
    
    batch_sizes = [32]
    learning_rates = [0.001]
    num_epochs = [50]
    num_layers = [1]
    d_models = [32]
    dropouts = [0.1]
    feedforward_dims = [64]
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for num_epoch in num_epochs:
                for num_layer in num_layers:
                    for d_model in d_models:
                        for dropout in dropouts:
                            for feedforward_dim in feedforward_dims:
                                train_and_test_model2(batch_size, learning_rate, num_epoch, window_len, forecast_len, input_size,
                                                     output_size, num_layer, dropout, feedforward_dim, train_dataset, test_dataset,
                                                     model_cls, loss_fn, optim_cls, train_fn, eval_fn, training_results_path,
                                                     predictions_path, model_type, d_model)

elif model_type == "decoder_v2":
    
    window_len = 365
    forecast_len = 30
    input_size = 5
    output_size = 1
    train_dataset = YahooDataset2(yahoo_dataset_path, window_len, forecast_len, train=True)
    scaler = train_dataset.get_scaler()
    test_dataset = YahooDataset2(yahoo_dataset_path, window_len, forecast_len, train=False, scaler=scaler)
    model_cls = TransformerDecoder_v2
    loss_fn = nn.MSELoss()
    optim_cls = optim.Adam
    train_fn = train_model
    eval_fn = eval_mae
    
    batch_sizes = [32]
    learning_rates = [0.001]
    num_epochs = [50]
    num_layers = [1]
    dropouts = [0.1]
    feedforward_dims = [64]
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for num_epoch in num_epochs:
                for num_layer in num_layers:
                    for dropout in dropouts:
                        for feedforward_dim in feedforward_dims:
                            train_and_test_model(batch_size, learning_rate, num_epoch, window_len, forecast_len, input_size,
                                                 output_size, num_layer, dropout, feedforward_dim, train_dataset, test_dataset,
                                                 model_cls, loss_fn, optim_cls, train_fn, eval_fn, training_results_path,
                                                 predictions_path, model_type)

elif model_type == "lstm":
    
    # TODO
    pass
