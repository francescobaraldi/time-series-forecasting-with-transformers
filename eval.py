import torch


def eval_mae(model, dl, device):
    cum_score = 0
    total = 0
    
    with torch.no_grad():
        for src, trg, class_idx in dl:
            class_idx = class_idx[0].item()
            model = model.to(device)
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            mae = torch.mean(torch.abs((out - trg[:, :, class_idx].unsqueeze(-1))))
            cum_score += mae
            total += 1
    
    return cum_score / total


def eval_mae2(model, dl, device):
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
            mae = torch.mean(torch.abs((out - trg[:, :, class_idx].unsqueeze(-1))))
            cum_score += mae
            total += 1
    
    return cum_score / total


def eval_mae_std(model, dl, device):
    cum_score = 0
    total = 0
    
    with torch.no_grad():
        for input, trg_y, class_idx in dl:
            class_idx = class_idx[0].item()
            model = model.to(device)
            src = input[:, :-1, :]
            trg = input[:, 1:, :]
            src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
            out = model(src, trg)
            mae = torch.mean(torch.abs((out - trg_y[:, :, class_idx].unsqueeze(-1))))
            cum_score += mae
            total += 1
    
    return cum_score / total
