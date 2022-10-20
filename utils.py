import torch

def reconstruct(scaler, trg, out):
    for b in range(out.shape[0]):
        out_rec = scaler.inverse_transform(out[b, :, :])
        trg_rec = scaler.inverse_transform(trg[b, :, :])
        out[b, :, :] = torch.from_numpy(out_rec)
        trg[b, :, :] = torch.from_numpy(trg_rec)
    
    return trg, out