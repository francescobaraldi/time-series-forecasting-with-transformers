import torch
from plot import plot_predictions


def test(device, model, dl, forecast_len, scaler, max_num=40, save_path=None):
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for j, (src, trg, class_idx) in enumerate(dl):
            if j == max_num:
                break
            class_idx = class_idx[0].item()
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            plot_predictions(src[0:1, :, :].cpu(), trg[0:1, :, :].cpu(), out[0:1, :, :].cpu(), scaler, forecast_len, class_idx.cpu(), j, save_path)
