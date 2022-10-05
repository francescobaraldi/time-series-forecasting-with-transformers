import torch


def eval_mae_singlestep(model, dl, device):
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


def eval_mae_multistep(model, dl, device):
    cum_score = 0
    total = 0
    
    with torch.no_grad():
        for input, trg, class_idx in dl:
            class_idx = class_idx[0].item()
            model = model.to(device)
            src = input[:, :-1, :]
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            mae = torch.mean(torch.abs((out - trg[:, :, class_idx].unsqueeze(-1))))
            cum_score += mae
            total += 1
    
    return cum_score / total
