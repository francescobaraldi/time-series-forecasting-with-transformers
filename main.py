import numpy as np
import torch
import joblib
import yfinance as yf

from model import StockTransformerDecoder, StockTransformer, StockLSTM
from inference import inference


def main():
    best_model_path = "best_models/"
    scaler_path = "weights/scaler_split_70.gz"
    inference_path = "inference_results/"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transformer_decoder_args = {
        'window_len': 90,
        'num_layers': 1,
        'input_size': 1,
        'output_size': 1,
        'd_model': 128,
        'num_heads': 8,
        'feedforward_dim': 256,
        'dropout': 0,
        'positional_encoding': 'sinusoidal',
    }
    transformer_args = {
        'window_len': 90,
        'target_len': 30,
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'input_size': 1,
        'output_size': 1,
        'd_model': 128,
        'num_heads': 8,
        'feedforward_dim': 256,
        'dropout': 0,
        'positional_encoding': 'sinusoidal',
    }
    lstm_args = {
        'input_size': 1,
        'hidden_dim': 64,
        'output_size': 1,
        'num_layers': 2,
        'dropout': 0,
    }
    
    transformer_decoder = StockTransformerDecoder(**transformer_decoder_args)
    transformer = StockTransformer(**transformer_args)
    lstm = StockLSTM(**lstm_args)
    transformer_decoder.load_state_dict(torch.load(best_model_path + "best_transformer_decoder.pth", map_location=torch.device(device)))
    transformer.load_state_dict(torch.load(best_model_path + "best_transformer.pth", map_location=torch.device(device)))
    lstm.load_state_dict(torch.load(best_model_path + "best_lstm.pth", map_location=torch.device(device)))
    
    # TODO
    
    # print("Forecasting the SP500 index closing price for the next 30 days...")
    
    # n = window_len + forecast_len - 1
    # data = yf.download('SPY').iloc[-n:]
    # data = data[['Close']].to_numpy()
    # scaler = joblib.load(scaler_path)
    # data_scaled = scaler.transform(data)
    # input = torch.from_numpy(data_scaled).unsqueeze(0)
    # inference(device=device, model=model, input=input, window_len=window_len, forecast_len=forecast_len, scaler=scaler,
    #           save_path=inference_path)
    # print(f"The prediction has been saved correctly in folder {inference_path}.")


if __name__ == "__main__":
    main()
