import torch
from plot import plot_inference


def inference(device, model, src, forecast_len, scaler, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        src_mae = 0
        src_mape = 0
        src = src.to(device)
        _, window_len, input_size = src.shape
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
            
            new_features = torch.cat((out[0, -1, :], torch.rand(4)))
            current_src = torch.cat((current_src[0:1, 1:, :], new_features.unsqueeze(0).unsqueeze(0)), dim=1)
            
        src_eval = src[0, :, :].clone().cpu()
        prediction_eval = prediction.clone().cpu()
        src_eval = scaler.inverse_transform(src_eval)[:, 0]
        prediction_eval = scaler.inverse_transform(prediction_eval + torch.zeros((window_len + forecast_len - 1, input_size)))[:, 0]
        src_eval = torch.from_numpy(src_eval)
        prediction_eval = torch.from_numpy(prediction_eval)
        src_mae += torch.mean(torch.abs(src_eval[1:] - prediction_eval[:window_len - 1]))
        src_mape += torch.mean(torch.abs((src_eval[1:] - prediction_eval[:window_len - 1]) / src_eval[1:]))
        
        plot_inference(src[0:1, :, :].cpu(), prediction.unsqueeze(0).cpu(), scaler, forecast_len, 0, save_path)
        
        return src_mae, src_mape