import torch.nn as nn
import torch.optim as optim

from dataset import YahooDataset
from model import TransformerDecoder, StockLSTM
from eval import eval_mae_singlestep, eval_mae_multistep, eval_mape_singlestep, eval_mape_multistep
from train import train_model_singlestep, train_model_multistep, train_and_test_model
from test import test_singlestep, test_multistep


yahoo_dataset_path = "datasets/yahoo_stock.csv"
predictions_path = "predictions/"
training_results_path = "training_results/"
weights_path = "weights/"

model_type = "transformer_decoder"

if model_type == "transformer_decoder":
    
    step_type = "singlestep"
    positional_encoding = "sinusoidal"
    
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    window_len = 365
    forecast_len = 60
    input_size = 5
    output_size = 1
    
    train_dataset = YahooDataset(dataset_path=yahoo_dataset_path, window_len=window_len, forecast_len=forecast_len,
                                 positional_encoding=positional_encoding, train=True)
    scaler = train_dataset.get_scaler()
    test_dataset = YahooDataset(dataset_path=yahoo_dataset_path, window_len=window_len, forecast_len=forecast_len,
                                positional_encoding=positional_encoding, train=False, scaler=scaler)
    
    model_cls = TransformerDecoder
    loss_fn = nn.MSELoss()
    optim_cls = optim.Adam
    train_fn = train_model_singlestep
    test_fn = test_singlestep
    eval_fn = eval_mae_singlestep
    
    num_layers = [1]
    d_models = [128]
    num_heads = [8]
    dropouts = [0]
    feedforward_dims = [512]
    for num_layer in num_layers:
        for d_model in d_models:
            for num_head in num_heads:
                for dropout in dropouts:
                    for feedforward_dim in feedforward_dims:
                        model_args = {
                            'window_len': window_len,
                            'num_layers': num_layer,
                            'input_size': input_size,
                            'output_size': output_size,
                            'd_model': d_model,
                            'num_heads': num_head,
                            'feedforward_dim': feedforward_dim,
                            'dropout': dropout,
                            'positional_encoding': positional_encoding,
                        }
                        train_and_test_model(batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs,
                                             forecast_len=forecast_len, train_dataset=train_dataset, test_dataset=test_dataset,
                                             model_cls=model_cls, loss_fn=loss_fn, optim_cls=optim_cls, train_fn=train_fn,
                                             test_fn=test_fn, eval_fn=eval_fn, training_results_path=training_results_path,
                                             predictions_path=predictions_path, weights_path=weights_path, model_type=model_type,
                                             step_type=step_type, model_args=model_args)

elif model_type == "lstm":
    
    step_type = "singlestep"
    
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    window_len = 365
    forecast_len = 60
    input_size = 5
    output_size = 1
    
    train_dataset = YahooDataset(dataset_path=yahoo_dataset_path, window_len=window_len, forecast_len=forecast_len,
                                 positional_encoding="learnable", train=True)
    scaler = train_dataset.get_scaler()
    test_dataset = YahooDataset(dataset_path=yahoo_dataset_path, window_len=window_len, forecast_len=forecast_len,
                                positional_encoding="learnable", train=False, scaler=scaler)
    
    model_cls = StockLSTM
    loss_fn = nn.MSELoss()
    optim_cls = optim.Adam
    train_fn = train_model_singlestep
    test_fn = test_singlestep
    eval_fn = eval_mae_singlestep
    
    num_layers = [2]
    hidden_dims = [128]
    dropouts = [0]
    for num_layer in num_layers:
        for hidden_dim in hidden_dims:
            for dropout in dropouts:
                model_args = {
                    'input_size': input_size,
                    'hidden_dim': hidden_dim,
                    'output_size': output_size,
                    'input_size': input_size,
                    'num_layers': num_layer,
                    'dropout': dropout
                }
                train_and_test_model(batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs,
                                     forecast_len=forecast_len, train_dataset=train_dataset, test_dataset=test_dataset,
                                     model_cls=model_cls, loss_fn=loss_fn, optim_cls=optim_cls, train_fn=train_fn, test_fn=test_fn,
                                     eval_fn=eval_fn, training_results_path=training_results_path,
                                     predictions_path=predictions_path, weights_path=weights_path, model_type=model_type,
                                     step_type=step_type, model_args=model_args)
