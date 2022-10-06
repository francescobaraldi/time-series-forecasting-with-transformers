import torch
import torch.nn as nn
from positional_encoding import SinusoidalPositionalEncoder, LearnablePositionalEncoder


class TransformerDecoder(nn.Module):
    def __init__(self, window_len, num_layers, input_size, output_size, d_model, num_heads, feedforward_dim, dropout=0.1,
                 positional_encoding="none"):
        super(TransformerDecoder, self).__init__()
        
        if positional_encoding == "sinusoidal":
            self.positional = SinusoidalPositionalEncoder(seq_len=window_len, d_model=d_model, dropout=dropout)
        elif positional_encoding == "learnable":
            self.positional = LearnablePositionalEncoder(seq_len=window_len, d_model=d_model, dropout=dropout)
        elif positional_encoding == "none":
            self.positional = None
        else:
            raise Exception("Positional encoding type not recognized: use 'none', 'sinusoidal' or 'learnable'.")
        
        self.window_len = window_len
        self.encode_input_layer = nn.Linear(input_size, d_model)
        encode_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=feedforward_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encode_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, output_size)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.encode_input_layer.weight.data.uniform_(-initrange, initrange)
        self.encode_input_layer.bias.data.zero_()
    
    def generate_mask(self, dim1, dim2):
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
        
    def forward(self, src, src_mask=None):
        src = self.encode_input_layer(src)
        if self.positional is not None:
            src = self.positional(src)
        
        if src_mask is None:
            src_mask = self.generate_mask(self.window_len, self.window_len).to(src.device)
        
        encoder_output = self.encoder(src, src_mask)
        output = self.output_layer(encoder_output)
        
        return output


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, dropout=0.1):
        super(StockLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.net(output)
        return output
