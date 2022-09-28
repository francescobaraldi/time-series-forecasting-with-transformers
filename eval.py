import torch


def eval_mae(model, dl, device):
    cum_score = 0
    total = 0
    
    with torch.no_grad():
        for src, trg, _ in dl:
            model = model.to(device)
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            mae = torch.mean(torch.abs((out - trg)))
            cum_score += mae
            total += 1
    
    return cum_score / total
