import numpy as np
import torch


def reconstruct(scaler, trg, out):
    for b in range(out.shape[0]):
        out_rec = scaler.inverse_transform(out[b, :, :])
        trg_rec = scaler.inverse_transform(trg[b, :, :])
        out[b, :, :] = torch.from_numpy(out_rec)
        trg[b, :, :] = torch.from_numpy(trg_rec)
    
    return trg, out


def eval_mae_singlestep(model, dl, device, scaler=None):
    name = "MAE"
    cum_score = 0
    total = 0
    
    with torch.no_grad():
        for input, _, class_idx in dl:
            class_idx = class_idx[0].item()
            model = model.to(device)
            src = input[:, :-1, :]
            trg = input[:, 1:, :]
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            if scaler is not None:
                trg, out = reconstruct(scaler, trg.cpu().numpy(), out.cpu().numpy())
                trg, out = torch.from_numpy(trg), torch.from_numpy(out)
            mae = torch.mean(torch.abs((out - trg)))
            cum_score += mae
            total += 1
    
    return (cum_score / total), name


def eval_mae_multistep(model, dl, device, scaler=None):
    name = "MAE"
    cum_score = 0
    total = 0
    
    with torch.no_grad():
        for input, trg, class_idx in dl:
            class_idx = class_idx[0].item()
            model = model.to(device)
            src = input[:, :-1, :]
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            if scaler is not None:
                trg, out = reconstruct(scaler, trg.cpu().numpy(), out.cpu().numpy())
                trg, out = torch.from_numpy(trg), torch.from_numpy(out)
            mae = torch.mean(torch.abs((out - trg)))
            cum_score += mae
            total += 1
    
    return (cum_score / total), name


def eval_mape_singlestep(model, dl, device, scaler=None):
    name = "MAPE"
    cum_score = 0
    total = 0
    
    with torch.no_grad():
        for input, _, class_idx in dl:
            class_idx = class_idx[0].item()
            model = model.to(device)
            src = input[:, :-1, :]
            trg = input[:, 1:, :]
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            if scaler is not None:
                trg, out = reconstruct(scaler, trg.cpu().numpy(), out.cpu().numpy())
                trg, out = torch.from_numpy(trg), torch.from_numpy(out)
            mape = torch.mean(torch.abs((out - trg) / trg))
            cum_score += mape
            total += 1
    
    return (cum_score / total), name


def eval_mape_multistep(model, dl, device, scaler=None):
    name = "MAPE"
    cum_score = 0
    total = 0
    
    with torch.no_grad():
        for input, trg, class_idx in dl:
            class_idx = class_idx[0].item()
            model = model.to(device)
            src = input[:, :-1, :]
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            if scaler is not None:
                trg, out = reconstruct(scaler, trg.cpu().numpy(), out.cpu().numpy())
                trg, out = torch.from_numpy(trg), torch.from_numpy(out)
            mape = torch.mean(torch.abs((out - trg) / trg))
            cum_score += mape
            total += 1
    
    return (cum_score / total), name


def eval_mae_std(model, dl, device):
    name = "MAE"
    cum_score = 0
    total = 0
    
    with torch.no_grad():
        for input, window_len, class_idx in dl:
            window_len = window_len[0].item()
            class_idx = class_idx[0].item()
            model = model.to(device)
            _, n, _ = input.shape
            forecast_len = n - window_len
            src = input[:, :window_len, :]
            trg = input[:, -forecast_len - 1:-1, :]
            trg_y = input[:, -forecast_len:, :]
            src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
            out = model(src, trg)
            mae = torch.mean(torch.abs((out - trg_y)))
            cum_score += mae
            total += 1
    
    return (cum_score / total), name
