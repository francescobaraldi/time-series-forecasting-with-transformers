import torch

def reconstruct(scaler, input):
    batch_size, _, _ = input.shape
    for b in range(batch_size):
        input_rec = scaler.inverse_transform(input[b, :, :])
        input[b, :, :] = torch.from_numpy(input_rec)
    
    return input