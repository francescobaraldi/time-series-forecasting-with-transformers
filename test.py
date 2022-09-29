import torch
from plot import plot_predictions, plot_predictions2


def test(device, model, dl, output_size, scaler, max_num=40, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for j, (src, trg, class_idx) in enumerate(dl):
            if j == max_num:
                break
            class_idx = class_idx[0].item()
            # src.shape = [batch_size, window_len, input_size]
            # trg.shape = [batch_size, window_len, input_size]
            # trg_forecast.shape = [batch_size, forecast_len, input_size]
            _, window_len, _ = src.shape
            _, forecast_len, _ = trg.shape
            n = window_len + forecast_len - 1
            predictions = torch.zeros((n, output_size))
            first = True
            current_src = src[0:1, :, :]
            for i in range(forecast_len):
                out = model(current_src)  # out.shape = [1, window_len, output_size]
                if first:
                    predictions[:window_len, :] = out[0, :, :]
                    first = False
                else:
                    predictions[i - 1 + window_len, :] = out[0, -1, :]
                new_features = torch.cat((out[0, -1, :], trg[0, 0, 1:]))
                current_src = torch.cat((current_src[0:1, 1:, :], new_features.unsqueeze(0).unsqueeze(0)), dim=1)
            
            plot_predictions(src, trg, predictions.unsqueeze(0), scaler, class_idx, j, save_path)


def test2(device, model, dl, forecast_len, scaler, max_num=40, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for j, (src, trg, class_idx) in enumerate(dl):
            if j == max_num:
                break
            class_idx = class_idx[0].item()
            # src.shape = [batch_size, window_len, input_size]
            # trg.shape = [batch_size, window_len, input_size]
            # trg_forecast.shape = [batch_size, forecast_len, input_size]
            out = model(src)
            plot_predictions2(src[0:1, :, :], trg[0:1, :, :], out[0:1, :, :], scaler, forecast_len, class_idx, j, save_path)
