import numpy as np
import torch
import joblib
import yfinance as yf

from model import TransformerDecoder
from inference import inference


def main():
    best_model_path = "best_model/best_model_weights.pth"
    scaler_path = "weights/scaler_split_50.gz"
    inference_path = "inference_results/"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    window_len = 365
    input_size = 5
    output_size = 1
    num_layers = 1
    d_model = 128
    num_heads = 8
    dropout = 0
    feedforward_dim = 512
    positional_encoding = "sinusoidal"
    
    model = TransformerDecoder(window_len=window_len, num_layers=num_layers, input_size=input_size, output_size=output_size,
                               d_model=d_model, num_heads=num_heads, feedforward_dim=feedforward_dim, dropout=dropout,
                               positional_encoding=positional_encoding)
    model.load_state_dict(torch.load(best_model_path))
    
    while True:
        forecast_len = input("How many days do you want to forecast? ")
        try:
            forecast_len = int(forecast_len)
            break
        except:
            print("Error: the value must be integer.")
    
    
    data = yf.download('SPY').iloc[-365:]
    data = data[['Close', 'High', 'Low', 'Open', 'Volume']].to_numpy()
    scaler = joblib.load(scaler_path)
    data_scaled = scaler.transform(data)
    src = torch.from_numpy(data_scaled).unsqueeze(0)
    src_mae, src_mape = inference(device=device, model=model, src=src, forecast_len=forecast_len, scaler=scaler,
                                  save_path=inference_path)
    print(f"The prediction has been saved correctly in folder {inference_path}.")


if __name__ == "__main__":
    main()
