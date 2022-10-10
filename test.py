import torch
from plot import plot_predictions


def test_singlestep(device, model, dl, forecast_len, scaler, max_num=50, save_path=None):
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
            batch_size, window_len, input_size = src.shape
            prediction = torch.zeros((batch_size, window_len + forecast_len - 1, input_size))
            first = True
            current_src = src
            for i in range(forecast_len):
                out = model(current_src)
                if first:
                    prediction[:, :window_len, :] = out
                    first = False
                else:
                    prediction[:, window_len + i - 1:, :] = out[:, -1:, :]
                
                current_src = torch.cat((current_src[:, 1:, :], out[:, -1:, :]), dim=1)
                
            src_eval = src.clone().cpu()
            trg_eval = trg[:, -forecast_len:, :].clone().cpu()
            prediction_eval = prediction.clone().cpu()
            # for b in range(batch_size):
            #     src_eval[b, :, :] = torch.from_numpy(scaler.inverse_transform(src_eval[b, :, :]))
            #     trg_eval[b, :, :] = torch.from_numpy(scaler.inverse_transform(trg_eval[b, :, :]))
            #     prediction_eval[b, :, :] = torch.from_numpy(scaler.inverse_transform(prediction_eval[b, :, :]))
            src_mae += torch.mean(torch.abs(src_eval[:, 1:, :] - prediction_eval[:, :window_len - 1, :]))
            src_mape += torch.mean(torch.abs((src_eval[:, 1:, :] - prediction_eval[:, :window_len - 1, :]) / src_eval[:, 1:, :]))
            trg_mae += torch.mean(torch.abs(trg_eval - prediction_eval[:, -forecast_len:, :]))
            trg_mape += torch.mean(torch.abs((trg_eval - prediction_eval[:, -forecast_len:, :]) / trg_eval))
            
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, -forecast_len:, :].cpu(), prediction[0:1, :, :].cpu(), scaler,
                              forecast_len, class_idx, j, save_path)
        
        src_mae /= count
        src_mape /= count
        trg_mae /= count
        trg_mape /= count
        
        return src_mae, src_mape, trg_mae, trg_mape


def test_multistep(device, model, dl, forecast_len, scaler, max_num=50, save_path=None):
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
            batch_size, window_len, input_size = src.shape
            class_idx = class_idx[0].item()
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            
            src_eval = src.clone().cpu()
            trg_eval = trg[:, -forecast_len:, :].clone().cpu()
            prediction_eval = out.clone().cpu()
            # for b in range(batch_size):
            #     src_eval[b, :, :] = torch.from_numpy(scaler.inverse_transform(src_eval[b, :, :]))
            #     trg_eval[b, :, :] = torch.from_numpy(scaler.inverse_transform(trg_eval[b, :, :]))
            #     prediction_eval[b, :, :] = torch.from_numpy(scaler.inverse_transform(prediction_eval[b, :, :]))
            src_mae += torch.mean(torch.abs(src_eval[:, forecast_len:, :] - prediction_eval[:, :window_len - forecast_len, :]))
            src_mape += torch.mean(torch.abs((src_eval[:, forecast_len:, :] - prediction_eval[:, :window_len - forecast_len, :]) / src_eval[:, forecast_len:, :]))
            trg_mae += torch.mean(torch.abs(trg_eval - prediction_eval[:, -forecast_len:, :]))
            trg_mape += torch.mean(torch.abs((trg_eval - prediction_eval[:, -forecast_len:, :]) / trg_eval))
            
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, :, :].cpu(), out[0:1, :, :].cpu(), scaler,
                             forecast_len, class_idx, j, save_path)
        
        src_mae /= count
        src_mape /= count
        trg_mae /= count
        trg_mape /= count
        
        return src_mae, src_mape, trg_mae, trg_mape


def test_std(device, model, dl, forecast_len, scaler, max_num=50, save_path=None):
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
            
            trg_y_eval = trg_y.clone().cpu()
            prediction_eval = out.clone().cpu()
            
            mae += torch.mean(torch.abs(trg_y_eval - prediction_eval))
            mape += torch.mean(torch.abs((trg_y_eval - prediction_eval) / trg_y_eval))
            
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, -forecast_len:, :].cpu(), out[0:1, :, :].cpu(), scaler,
                             forecast_len, class_idx, j, save_path)
        
        mae /= count
        mape /= count
        
        return mae, mape
