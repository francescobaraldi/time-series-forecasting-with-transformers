import torch
from plot import plot_inference


def inference(device, model, input, window_len, forecast_len, scaler, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        mae = 0
        mape = 0
        _, _, input_size = input.shape
        src = input[:, :window_len, :]
        trg = input[:, -forecast_len:, :]
        src, trg = src.to(device), trg.to(device)
        prediction = torch.zeros((forecast_len, input_size))
        current_src = src
        current_trg = trg
        src_idx = 0
        for i in range(forecast_len):
            out = model(current_src, current_trg)
            prediction[i, :] = out[0, -1, :]
            
            #TODO
            current_src = input[:, src_idx:src_idx + window_len]
            current_trg = torch.cat((current_trg[0:1, 1:, :], prediction[i, :].unsqueeze(0).unsqueeze(0)), dim=1)
            src_idx += 1
            
        src_eval = src[0, :, :].clone().cpu()
        prediction_eval = prediction.clone().cpu()
        src_eval = scaler.inverse_transform(src_eval)[:, 0]
        prediction_eval = scaler.inverse_transform(prediction_eval + torch.zeros((window_len + forecast_len - 1, input_size)))[:, 0]
        src_eval = torch.from_numpy(src_eval)
        prediction_eval = torch.from_numpy(prediction_eval)
        mae += torch.mean(torch.abs(src_eval[1:] - prediction_eval[:window_len - 1]))
        mape += torch.mean(torch.abs((src_eval[1:] - prediction_eval[:window_len - 1]) / src_eval[1:]))
        
        plot_inference(src[0:1, :, :].cpu(), prediction.unsqueeze(0).cpu(), scaler, forecast_len, 0, save_path)
        
        return mae, mape