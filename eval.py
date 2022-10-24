import torch


def eval_transformer_decoder(model, dl, device, eval_name="mae"):
    cum_score = 0
    total = 0
    
    if eval_name != "mae" and eval_name != "mape":
        raise Exception("Eval function not recognized: use 'mae' or 'mape'.")
    
    with torch.no_grad():
        for input, window_len, class_idx in dl:
            window_len = window_len[0].item()
            class_idx = class_idx[0].item()
            model = model.to(device)
            src = input[:, :window_len, :]
            trg = input[:, 1:1 + window_len, :]
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            if eval_name == "mae":
                score = torch.mean(torch.abs(out - trg))
            elif eval_name == "mape":
                score = torch.mean(torch.abs((out - trg) / trg))
            cum_score += score
            total += 1
    
    return (cum_score / total)


def eval_transformer(model, dl, device, eval_name="mae"):
    cum_score = 0
    total = 0
    
    if eval_name != "mae" and eval_name != "mape":
        raise Exception("Eval function not recognized: use 'mae' or 'mape'.")
    
    with torch.no_grad():
        for input, window_len, class_idx in dl:
            window_len = window_len[0].item()
            class_idx = class_idx[0].item()
            model = model.to(device)
            _, n, _ = input.shape
            forecast_len = (n - window_len + 1) // 2
            src = input[:, :window_len, :]
            trg = input[:, window_len - 1:window_len - 1 + forecast_len, :]
            trg_y = input[:, window_len:window_len + forecast_len, :]
            src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
            out = model(src, trg)
            if eval_name == "mae":
                score = torch.mean(torch.abs(out - trg_y))
            elif eval_name == "mape":
                score = torch.mean(torch.abs((out - trg_y) / trg_y))
            cum_score += score
            total += 1
    
    return (cum_score / total)


def eval_lstm(model, dl, device, eval_name="mae"):
    cum_score = 0
    total = 0
    
    if eval_name != "mae" and eval_name != "mape":
        raise Exception("Eval function not recognized: use 'mae' or 'mape'.")
    
    with torch.no_grad():
        for input, window_len, class_idx in dl:
            window_len = window_len[0].item()
            class_idx = class_idx[0].item()
            model = model.to(device)
            src = input[:, :window_len, :]
            trg = input[:, window_len:window_len + 1, :]
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            if eval_name == "mae":
                score = torch.mean(torch.abs(out - trg))
            elif eval_name == "mape":
                score = torch.mean(torch.abs((out - trg) / trg))
            cum_score += score
            total += 1
    
    return (cum_score / total)
