import torch
from plot import plot_predictions
from utils import reconstruct


def test_transformer_decoder(device, model, dl, forecast_len, scaler, max_num=50, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        mae = 0
        mape = 0
        count = 0
        for j, (input, window_len, class_idx) in enumerate(dl):
            if j >= max_num:
                break
            
            count += 1
            window_len = window_len[0].item()
            class_idx = class_idx[0].item()
            src = input[:, :window_len, :]
            trg = input[:, window_len:window_len + forecast_len, :]
            src, trg = src.to(device), trg.to(device)
            batch_size, _, input_size = src.shape
            prediction = torch.zeros((batch_size, forecast_len, input_size)).to(device)
            current_src = src
            for i in range(forecast_len):
                out = model(current_src)
                prediction[:, i, :] = out[:, -1, :]
                
                current_src = torch.cat((current_src[:, 1:, :], out[:, -1:, :]), dim=1)
            
            mae += torch.mean(torch.abs(trg - prediction))
            mape += torch.mean(torch.abs((trg - prediction) / trg))
            
            if scaler is not None:
                src = reconstruct(scaler, src.cpu())
                trg = reconstruct(scaler, trg.cpu())
                prediction = reconstruct(scaler, prediction.cpu())
            plot_predictions(src[0, :, :], trg[0, :, :], prediction[0, :, :], forecast_len, class_idx, j, save_path)
        
        mae /= count
        mape /= count
    
        return mae, mape


def test_transformer(device, model, dl, forecast_len, scaler, max_num=50, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        mae = 0
        mape = 0
        count = 0
        for j, (input, window_len, class_idx) in enumerate(dl):
            if j >= max_num:
                break
            
            count += 1
            window_len = window_len[0].item()
            class_idx = class_idx[0].item()
            input = input.to(device)
            src = input[:, :window_len, :]
            trg = input[:, window_len - 1:window_len - 1 + forecast_len, :]
            trg_y = input[:, -forecast_len:, :]
            batch_size, _, input_size = src.shape
            prediction = torch.zeros((batch_size, forecast_len, input_size)).to(device)
            current_src = src
            current_trg = trg
            for i in range(forecast_len):
                out = model(current_src, current_trg)
                prediction[:, i, :] = out[:, -1, :]
                src_idx = i + 1
                current_src = input[:, src_idx:src_idx + window_len]
                current_trg = torch.cat((current_trg[:, 1:, :], prediction[:, i, :].unsqueeze(1)), dim=1)
            
            mae += torch.mean(torch.abs(trg_y - prediction))
            mape += torch.mean(torch.abs((trg_y - prediction) / trg_y))
            
            if scaler is not None:
                current_src = reconstruct(scaler, current_src.cpu())
                trg_y = reconstruct(scaler, trg_y.cpu())
                prediction = reconstruct(scaler, prediction.cpu())
            plot_predictions(current_src[0, :, :], trg_y[0, :, :], prediction[0, :, :], forecast_len, class_idx, j, save_path)
        
        mae /= count
        mape /= count
        
        return mae, mape


def test_lstm(device, model, dl, forecast_len, scaler, max_num=50, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        mae = 0
        mape = 0
        count = 0
        for j, (input, window_len, class_idx) in enumerate(dl):
            if j >= max_num:
                break
            
            count += 1
            window_len = window_len[0].item()
            class_idx = class_idx[0].item()
            src = input[:, :window_len, :]
            trg = input[:, window_len:window_len + forecast_len, :]
            src, trg = src.to(device), trg.to(device)
            batch_size, _, input_size = src.shape
            prediction = torch.zeros((batch_size, forecast_len, input_size)).to(device)
            current_src = src
            for i in range(forecast_len):
                out = model(current_src)
                prediction[:, i, :] = out[:, -1, :]
                
                current_src = torch.cat((current_src[:, 1:, :], out[:, -1:, :]), dim=1)
            
            mae += torch.mean(torch.abs(trg - prediction))
            mape += torch.mean(torch.abs((trg - prediction) / trg))
            
            if scaler is not None:
                src = reconstruct(scaler, src.cpu())
                trg = reconstruct(scaler, trg.cpu())
                prediction = reconstruct(scaler, prediction.cpu())
            plot_predictions(src[0, :, :], trg[0, :, :], prediction[0, :, :], forecast_len, class_idx, j, save_path)
        
        mae /= count
        mape /= count
        
        return mae, mape
