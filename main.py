import torch
import joblib
import yfinance as yf

from model import StockTransformerDecoder, StockTransformer, StockLSTM
from inference import inference_transformer_decoder, inference_transformer, inference_lstm


def main():
    best_model_path = "best_models/"
    scaler_path = "weights/scaler_split_80.gz"
    inference_path = "inference_results/"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    window_len = 90
    forecast_len = 30
    
    transformer_decoder_args = {
        'window_len': window_len,
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
        'window_len': window_len,
        'target_len': forecast_len,
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
    
    scaler = joblib.load(scaler_path)
    gspc = yf.Ticker("^GSPC")
    gspc = gspc.history(period="1y")
    
    print("\nForecasting the SP500 index closing price for the next 30 days with transformer decoder model...")
    
    data = gspc[['Close']].iloc[-window_len:].to_numpy()
    scaler = joblib.load(scaler_path)
    data_scaled = scaler.transform(data)
    src = torch.from_numpy(data_scaled).float().unsqueeze(0)
    inference_transformer_decoder(device=device, model=transformer_decoder, src=src, forecast_len=forecast_len, scaler=scaler,
                                  save_path=inference_path + "prediction_transformer_decoder.png")
    
    print(f"The prediction of the transformer decoder model has been saved correctly in folder {inference_path}")
    
    
    print("\nForecasting the SP500 index closing price for the next 30 days with transformer model...")
    
    n = window_len + forecast_len - 1
    data = gspc[['Close']].iloc[-n:].to_numpy()
    scaler = joblib.load(scaler_path)
    data_scaled = scaler.transform(data)
    input = torch.from_numpy(data_scaled).float().unsqueeze(0)
    inference_transformer(device=device, model=transformer, input=input, window_len=window_len, forecast_len=forecast_len,
                          scaler=scaler, save_path=inference_path + "prediction_transformer.png")
    
    print(f"The prediction of the transformer model has been saved correctly in folder {inference_path}")
    
    
    print("\nForecasting the SP500 index closing price for the next 30 days with lstm model...")
    
    data = gspc[['Close']].iloc[-window_len:].to_numpy()
    scaler = joblib.load(scaler_path)
    data_scaled = scaler.transform(data)
    src = torch.from_numpy(data_scaled).float().unsqueeze(0)
    inference_lstm(device=device, model=lstm, src=src, forecast_len=forecast_len, scaler=scaler,
                   save_path=inference_path + "prediction_lstm.png")
    
    print(f"The prediction of the lstm model has been saved correctly in folder {inference_path}")


if __name__ == "__main__":
    main()
