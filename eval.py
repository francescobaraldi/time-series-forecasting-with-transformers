import numpy as np
import torch


def reconstruct(scaler, trg, out):
    for b in range(out.shape[0]):
        add = np.zeros(out.shape[1], trg.shape[2])
        out_rec = scaler.inverse_transform(out[b, :, :] + add)
        trg_rec = scaler.inverse_transform(trg[b, :, :])
        out[b, :, 0] = torch.from_numpy(out_rec[:, 0])
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
            mae = torch.mean(torch.abs((out - trg[:, :, class_idx].unsqueeze(-1))))
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
            mae = torch.mean(torch.abs((out - trg[:, :, class_idx].unsqueeze(-1))))
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
            mape = torch.mean(torch.abs((out - trg[:, :, class_idx].unsqueeze(-1)) / trg[:, :, class_idx].unsqueeze(-1)))
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
            mape = torch.mean(torch.abs((out - trg[:, :, class_idx].unsqueeze(-1)) / trg[:, :, class_idx].unsqueeze(-1)))
            cum_score += mape
            total += 1
    
    return (cum_score / total), name
