import torch
from plot import plot_predictions, plot_predictions2


def test(device, model, dl, forecast_len, scaler, max_num=40, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for j, (src, trg, class_idx) in enumerate(dl):
            if j == max_num:
                break
            class_idx = class_idx[0].item()
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, :, :].cpu(), out[0:1, :, :].cpu(), scaler, forecast_len, class_idx, j, save_path)


def test2(device, model, dl, forecast_len, scaler, max_num=40, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for j, (input, trg, class_idx) in enumerate(dl):
            if j == max_num:
                break
            
            class_idx = class_idx[0].item()
            src = input[:, :-1, :]
            src, trg = src.to(device), trg.to(device)
            _, window_len, _ = src.shape
            prediction = torch.zeros((window_len + forecast_len - 1, trg.shape[2]))
            first = True
            current_src = src
            for i in range(forecast_len):
                out = model(current_src)
                if first:
                    prediction[:window_len, :] = out[0, :, :]
                    first = False
                else:
                    prediction[window_len + i - 1:, :] = out[0, -1, :]
                
                new_features = torch.cat((out[0, -1, :], trg[0, i, 1:]))
                current_src = torch.cat((current_src[0:1, 1:, :], new_features.unsqueeze(0).unsqueeze(0)), dim=1)
            plot_predictions2(src[0:1, :, :].cpu(), trg[0:1, :, :].cpu(), prediction[:, :].unsqueeze(0).cpu(), scaler, forecast_len, class_idx, j, save_path)


def test_std(device, model, dl, forecast_len, scaler, max_num=40, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for j, (input, trg_y, class_idx) in enumerate(dl):
            if j == max_num:
                break
            
            class_idx = class_idx[0].item()
            src = input[:, :-1, :]
            trg = input[:, 1:, :]
            src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
            trg_y = trg_y[:, -forecast_len, :]
            _, window_len, _ = src.shape
            prediction = torch.zeros((window_len + forecast_len - 1, trg.shape[2]))
            first = True
            current_src = src
            current_trg = trg
            for i in range(forecast_len):
                out = model(current_src, current_trg)
                if first:
                    prediction[:window_len, :] = out[0, :, :]
                    first = False
                else:
                    prediction[window_len + i - 1:, :] = out[0, -1, :]
                
                new_features = torch.cat((out[0, -1, :], trg_y[0, i, 1:]))
                current_src = torch.cat((current_src[0:1, 1:, :], new_features.unsqueeze(0).unsqueeze(0)), dim=1)
                current_trg = torch.cat((current_trg[0:1, 1:, :], new_features.unsqueeze(0).unsqueeze(0)), dim=1)
            plot_predictions2(src[0:1, :, :].cpu(), trg_y[0:1, :, :].cpu(), prediction[:, :].unsqueeze(0).cpu(), scaler, forecast_len, class_idx, j, save_path)
