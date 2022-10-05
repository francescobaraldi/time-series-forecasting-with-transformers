import torch
from plot import plot_predictions


def test_singlestep(device, model, dl, forecast_len, scaler, max_num=40, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for j, (input, trg, class_idx) in enumerate(dl):
            if j >= max_num:
                break
            
            class_idx = class_idx[0].item()
            src = input[:, :-1, :]
            src, trg = src.to(device), trg.to(device)
            _, window_len, _ = src.shape
            prediction = torch.zeros((window_len + forecast_len - 1, 1))
            first = True
            current_src = src
            for i in range(forecast_len):
                out = model(current_src)
                if first:
                    prediction[:window_len, :] = out[0, :, :]
                    first = False
                else:
                    prediction[window_len + i - 1:, :] = out[0, -1, :]
                
                new_features = torch.cat((out[0, -1, :], trg[0, -(forecast_len - i), 1:]))
                current_src = torch.cat((current_src[0:1, 1:, :], new_features.unsqueeze(0).unsqueeze(0)), dim=1)
            
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, -forecast_len:, :].cpu(), prediction.unsqueeze(0).cpu(), scaler,
                              forecast_len, class_idx, j, save_path)


def test_multistep(device, model, dl, forecast_len, scaler, max_num=40, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for j, (input, trg, class_idx) in enumerate(dl):
            if j >= max_num:
                break
            
            src = input[:, :-1, :]
            class_idx = class_idx[0].item()
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, -forecast_len:, :].cpu(), out[0:1, :, :].cpu(), scaler,
                             forecast_len, class_idx, j, save_path)
