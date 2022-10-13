import torch
from plot import plot_predictions


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
            trg = input[:, -forecast_len:, :]
            src, trg = src.to(device), trg.to(device)
            batch_size, _, input_size = src.shape
            prediction = torch.zeros((batch_size, forecast_len, input_size))
            current_src = src
            for i in range(forecast_len):
                out = model(current_src)
                prediction[:, i, :] = out[:, -1, :]
                
                current_src = torch.cat((current_src[:, 1:, :], out[:, -1:, :]), dim=1)
                
            mae += torch.mean(torch.abs(trg - prediction))
            mape += torch.mean(torch.abs((trg - prediction) / trg))
            
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, :, :].cpu(), prediction[0:1, :, :].cpu(), scaler, forecast_len, class_idx,
                             j, save_path)
        
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
            src = input[:, :window_len, :]
            trg = input[:, -forecast_len - 1:-1, :]
            trg_y = input[:, -forecast_len:, :]
            src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
            out = model(src, trg)
            
            mae += torch.mean(torch.abs(trg_y - out))
            mape += torch.mean(torch.abs((trg_y - out) / trg_y))
            
            plot_predictions(src[0:1, :, :].cpu(), trg_y[0:1, :, :].cpu(), out[0:1, :, :].cpu(), scaler, forecast_len, class_idx, j,
                             save_path)
        
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
            trg = input[:, -forecast_len:, :]
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            
            mae += torch.mean(torch.abs(trg - out))
            mape += torch.mean(torch.abs((trg - out) / trg))
            
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, :, :].cpu(), out[0:1, :, :].cpu(), scaler, forecast_len, class_idx, j,
                             save_path)
        
        mae /= count
        mape /= count
        
        return mae, mape
