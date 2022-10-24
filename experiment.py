import torch.nn as nn
import torch.optim as optim
import joblib

from dataset import YahooDataset
from model import StockTransformerDecoder, StockTransformer, StockLSTM
from eval import eval_transformer_decoder, eval_transformer, eval_lstm
from train import train_transformer_decoder, train_transformer, train_lstm, train_and_test_model
from test import test_transformer_decoder, test_transformer, test_lstm


yahoo_dataset_path = "datasets/sp500.csv"
predictions_path = "predictions/"
training_results_path = "training_results/"
weights_path = "weights/"

models_dict = {
    'transformer_decoder': {
        'model_cls': StockTransformerDecoder,
        'train_fn': train_transformer_decoder,
        'test_fn': test_transformer_decoder,
        'eval_fn': eval_transformer_decoder,
    },
    'transformer': {
        'model_cls': StockTransformer,
        'train_fn': train_transformer,
        'test_fn': test_transformer,
        'eval_fn': eval_transformer,
    },
    'lstm': {
        'model_cls': StockLSTM,
        'train_fn': train_lstm,
        'test_fn': test_lstm,
        'eval_fn': eval_lstm,
    },
}

model_type = "transformer"

if model_type == "transformer_decoder":
    
    positional_encoding = "sinusoidal"
    
    num_epochs = 35
    batch_size = 64
    learning_rate = 0.5e-05
    weight_decay = 1e-06
    window_len = 90
    forecast_len = 30
    input_size = 1
    output_size = 1
    
    train_rate = 0.8
    train_dataset = YahooDataset(dataset_path=yahoo_dataset_path, window_len=window_len, forecast_len=forecast_len, train=True,
                                 train_rate=train_rate, scalertype="minmax")
    scaler = train_dataset.get_scaler()
    joblib.dump(scaler, f"{weights_path}scaler_split_{int(train_rate*100)}.gz")
    test_dataset = YahooDataset(dataset_path=yahoo_dataset_path, window_len=window_len, forecast_len=forecast_len, train=False,
                                train_rate=train_rate, scaler=scaler)
    
    loss_fn = nn.MSELoss()
    optim_cls = optim.Adam
    eval_name = "mae"
    model_dict = models_dict[model_type]
    
    num_layers = [1]
    d_models = [128]
    num_heads = [8]
    dropouts = [0.03]
    feedforward_dims = [256]
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
                        train_and_test_model(batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
                                             num_epochs=num_epochs, forecast_len=forecast_len, train_dataset=train_dataset,
                                             test_dataset=test_dataset, model_cls=model_dict['model_cls'], loss_fn=loss_fn,
                                             optim_cls=optim_cls, train_fn=model_dict['train_fn'], test_fn=model_dict['test_fn'],
                                             eval_fn=model_dict['eval_fn'], training_results_path=training_results_path,
                                             predictions_path=predictions_path, weights_path=weights_path, model_type=model_type,
                                             eval_name=eval_name, model_args=model_args)

elif model_type == "transformer":
    
    positional_encoding = "learnable"
    
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.5e-05
    weight_decay = 1e-06
    window_len = 90
    forecast_len = 30
    input_size = 1
    output_size = 1
    
    train_rate = 0.8
    train_dataset = YahooDataset(dataset_path=yahoo_dataset_path, window_len=window_len, forecast_len=forecast_len, train=True,
                                 train_rate=train_rate, scalertype="minmax")
    scaler = train_dataset.get_scaler()
    joblib.dump(scaler, f"{weights_path}scaler_split_{int(train_rate*100)}.gz")
    test_dataset = YahooDataset(dataset_path=yahoo_dataset_path, window_len=window_len, forecast_len=forecast_len, train=False,
                                train_rate=train_rate, scaler=scaler)
    
    loss_fn = nn.MSELoss()
    optim_cls = optim.Adam
    eval_name = "mae"
    model_dict = models_dict[model_type]
    
    num_layers = [1]
    d_models = [128]
    num_heads = [8]
    dropouts = [0.08]
    feedforward_dims = [256]
    for num_layer in num_layers:
        for d_model in d_models:
            for num_head in num_heads:
                for dropout in dropouts:
                    for feedforward_dim in feedforward_dims:
                        model_args = {
                            'window_len': window_len,
                            'target_len': forecast_len,
                            'num_encoder_layers': num_layer,
                            'num_decoder_layers': num_layer,
                            'input_size': input_size,
                            'output_size': output_size,
                            'd_model': d_model,
                            'num_heads': num_head,
                            'feedforward_dim': feedforward_dim,
                            'dropout': dropout,
                            'positional_encoding': positional_encoding,
                        }
                        train_and_test_model(batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
                                             num_epochs=num_epochs, forecast_len=forecast_len, train_dataset=train_dataset,
                                             test_dataset=test_dataset, model_cls=model_dict['model_cls'], loss_fn=loss_fn,
                                             optim_cls=optim_cls, train_fn=model_dict['train_fn'], test_fn=model_dict['test_fn'],
                                             eval_fn=model_dict['eval_fn'], training_results_path=training_results_path,
                                             predictions_path=predictions_path, weights_path=weights_path, model_type=model_type,
                                             eval_name=eval_name, model_args=model_args)

elif model_type == "lstm":
    
    num_epochs = 20
    batch_size = 64
    learning_rate = 1e-05
    weight_decay = 1e-05
    window_len = 90
    forecast_len = 30
    input_size = 1
    output_size = 1
    
    train_rate = 0.8
    train_dataset = YahooDataset(dataset_path=yahoo_dataset_path, window_len=window_len, forecast_len=forecast_len, train=True,
                                 train_rate=train_rate, scalertype="minmax")
    scaler = train_dataset.get_scaler()
    joblib.dump(scaler, f"{weights_path}scaler_split_{int(train_rate*100)}.gz")
    test_dataset = YahooDataset(dataset_path=yahoo_dataset_path, window_len=window_len, forecast_len=forecast_len, train=False,
                                train_rate=train_rate, scaler=scaler)
    
    loss_fn = nn.MSELoss()
    optim_cls = optim.Adam
    eval_name = "mae"
    model_dict = models_dict[model_type]
    
    num_layers = [2]
    hidden_dims = [64]
    dropouts = [0.05]
    for num_layer in num_layers:
        for hidden_dim in hidden_dims:
            for dropout in dropouts:
                model_args = {
                    'input_size': input_size,
                    'hidden_dim': hidden_dim,
                    'output_size': output_size,
                    'num_layers': num_layer,
                    'dropout': dropout
                }
                train_and_test_model(batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
                                     num_epochs=num_epochs, forecast_len=forecast_len, train_dataset=train_dataset,
                                     test_dataset=test_dataset, model_cls=model_dict['model_cls'], loss_fn=loss_fn,
                                     optim_cls=optim_cls, train_fn=model_dict['train_fn'], test_fn=model_dict['test_fn'],
                                     eval_fn=model_dict['eval_fn'], training_results_path=training_results_path,
                                     predictions_path=predictions_path, weights_path=weights_path, model_type=model_type,
                                     eval_name=eval_name, model_args=model_args)
