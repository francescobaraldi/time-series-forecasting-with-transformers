import torch
from plot import plot_predictions


def test_singlestep(device, model, dl, forecast_len, scaler, max_num=40, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        src_mae = 0
        src_mape = 0
        trg_mae = 0
        trg_mape = 0
        count = 0
        for j, (input, trg, class_idx) in enumerate(dl):
            if j >= max_num:
                break
            
            count += 1
            class_idx = class_idx[0].item()
            src = input[:, :-1, :]
            src, trg = src.to(device), trg.to(device)
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
                
                new_features = torch.cat((out[0, -1, :], trg[0, -(forecast_len - i), 1:]))
                current_src = torch.cat((current_src[0:1, 1:, :], new_features.unsqueeze(0).unsqueeze(0)), dim=1)
                
            src_eval = src[0, :, :].clone().cpu()
            trg_eval = trg[0, -forecast_len:, :].clone().cpu()
            prediction_eval = prediction.clone().cpu()
            src_eval = scaler.inverse_transform(src_eval)[:, class_idx]
            trg_eval = scaler.inverse_transform(trg_eval)[:, class_idx]
            prediction_eval = scaler.inverse_transform(prediction_eval + torch.zeros((window_len + forecast_len - 1, input_size)))[:, class_idx]
            src_eval = torch.from_numpy(src_eval)
            trg_eval = torch.from_numpy(trg_eval)
            prediction_eval = torch.from_numpy(prediction_eval)
            src_mae += torch.mean(torch.abs(src_eval[1:] - prediction_eval[:window_len - 1]))
            src_mape += torch.mean(torch.abs((src_eval[1:] - prediction_eval[:window_len - 1]) / src_eval[1:]))
            trg_mae += torch.mean(torch.abs(trg_eval - prediction_eval[-forecast_len:]))
            trg_mape += torch.mean(torch.abs((trg_eval - prediction_eval[-forecast_len:]) / trg_eval))
            
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, -forecast_len:, :].cpu(), prediction.unsqueeze(0).cpu(), scaler,
                              forecast_len, class_idx, j, save_path)
        
        src_mae /= count
        src_mape /= count
        trg_mae /= count
        trg_mape /= count
        
        return src_mae, src_mape, trg_mae, trg_mape


def test_multistep(device, model, dl, forecast_len, scaler, max_num=40, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        src_mae = 0
        src_mape = 0
        trg_mae = 0
        trg_mape = 0
        count = 0
        for j, (input, trg, class_idx) in enumerate(dl):
            if j >= max_num:
                break
            
            count += 1
            src = input[:, :-1, :]
            _, window_len, input_size = src.shape
            class_idx = class_idx[0].item()
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            
            src_eval = src[0, :, :].clone().cpu()
            trg_eval = trg[0, -forecast_len:, :].clone().cpu()
            prediction_eval = out[0, :, :].clone().cpu()
            src_eval = scaler.inverse_transform(src_eval)[:, class_idx]
            trg_eval = scaler.inverse_transform(trg_eval)[:, class_idx]
            prediction_eval = scaler.inverse_transform(prediction_eval + torch.zeros((window_len, input_size)))[:, class_idx]
            src_eval = torch.from_numpy(src_eval)
            trg_eval = torch.from_numpy(trg_eval)
            prediction_eval = torch.from_numpy(prediction_eval)
            src_mae += torch.mean(torch.abs(src_eval[forecast_len:] - prediction_eval[:window_len - forecast_len]))
            src_mape += torch.mean(torch.abs((src_eval[forecast_len:] - prediction_eval[:window_len - forecast_len]) / src_eval[forecast_len:]))
            trg_mae += torch.mean(torch.abs(trg_eval - prediction_eval[-forecast_len:]))
            trg_mape += torch.mean(torch.abs((trg_eval - prediction_eval[-forecast_len:]) / trg_eval))
            
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, -forecast_len:, :].cpu(), out[0:1, :, :].cpu(), scaler,
                             forecast_len, class_idx, j, save_path)
        
        src_mae /= count
        src_mape /= count
        trg_mae /= count
        trg_mape /= count
        
        return src_mae, src_mape, trg_mae, trg_mape
