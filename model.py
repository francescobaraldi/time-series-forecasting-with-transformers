import torch
import torch.nn as nn
from tqdm import tqdm


def eval(model, dl, device):
    total_error = 0
    total = 0
    
    with torch.no_grad():
        for seq, trg in tqdm(dl):
            seq, trg = seq.to(device), trg.to(device)
            seq_mask = torch.ones_like(seq)
            out = model(seq, seq_mask)
            total_error += torch.sum(torch.abs(out - trg))
            total += out.size(0)
    
    return total_error / total
            


class Transformer(nn.Module):
    def __init__(self, seq_len, num_encoder, input_size, embed_dim, num_heads, feedforward_dim):
        super(Transformer, self).__init__()
        self.linear1 = nn.Linear(input_size, embed_dim)
        encode_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=feedforward_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encode_layer, num_encoder)
        # self.positional = nn.parameter.Parameter(torch.zeros(seq_len))
        self.linear2 = nn.Linear(embed_dim, seq_len)
        self.seq_len = seq_len
        
    def forward(self, seq, seq_mask):
        assert(seq.shape[1] == seq_mask.shape[1] == self.seq_len)
        # pos = self.positional + seq
        seq_mapped = self.linear1(seq)
        out = self.encoder(seq_mapped)
        out = self.linear2(out)
        return out
           

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()
        
    def forward(self, queries, keys, values):
        import math
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = torch.nn.functional.softmax(scores)
        out = torch.bmm(self.attention_weights, values)
        return out
