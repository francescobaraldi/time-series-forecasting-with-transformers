import torch
from plot import plot_inference


def inference_transformer_decoder(device, model, src, forecast_len, scaler, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        _, _, input_size = src.shape
        src = src.to(device)
        prediction = torch.zeros((forecast_len, input_size)).to(device)
        current_src = src
        for i in range(forecast_len):
            out = model(current_src)
            prediction[i, :] = out[0, -1, :]
            current_src = torch.cat((current_src[:, 1:, :], out[:, -1:, :]), dim=1)
            
        plot_inference(src[0, :, :].cpu(), prediction.cpu(), scaler, forecast_len, 0, save_path)


def inference_transformer(device, model, input, window_len, forecast_len, scaler, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        _, _, input_size = input.shape
        src = input[:, :window_len, :]
        trg = input[:, -forecast_len:, :]
        src, trg = src.to(device), trg.to(device)
        prediction = torch.zeros((forecast_len, input_size)).to(device)
        current_src = src
        current_trg = trg
        for i in range(forecast_len):
            out = model(current_src, current_trg)
            prediction[i, :] = out[0, -1, :]
            src_idx = i + 1
            current_src = input[:, src_idx:src_idx + window_len]
            current_trg = torch.cat((current_trg[0:1, 1:, :], prediction[i, :].unsqueeze(0).unsqueeze(0)), dim=1)
            
        plot_inference(src[0, :, :].cpu(), prediction.cpu(), scaler, forecast_len, 0, save_path)


def inference_lstm(device, model, src, forecast_len, scaler, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        _, _, input_size = src.shape
        src = src.to(device)
        prediction = torch.zeros((forecast_len, input_size))
        current_src = src
        for i in range(forecast_len):
            out = model(current_src)
            prediction[i, :] = out[0, -1, :]
            current_src = torch.cat((current_src[:, 1:, :], out[:, -1:, :]), dim=1)
            
        plot_inference(src[0, :, :].cpu(), prediction.cpu(), scaler, forecast_len, 0, save_path)
