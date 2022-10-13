import torch
import torch.nn as nn
from positional_encoding import SinusoidalPositionalEncoder, LearnablePositionalEncoder


class StockTransformerDecoder(nn.Module):
    def __init__(self, window_len, num_layers, input_size, output_size, d_model, num_heads, feedforward_dim, dropout=0.1,
                 positional_encoding="sinusoidal"):
        super(StockTransformerDecoder, self).__init__()
        
        if positional_encoding == "sinusoidal":
            self.positional = SinusoidalPositionalEncoder(seq_len=window_len, d_model=d_model, dropout=dropout)
        elif positional_encoding == "learnable":
            self.positional = LearnablePositionalEncoder(seq_len=window_len, d_model=d_model, dropout=dropout)
        else:
            raise Exception("Positional encoding type not recognized: use 'sinusoidal' or 'learnable'.")
        self.positional_encoding = positional_encoding
        
        self.window_len = window_len
        self.encoder_input_layer = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=feedforward_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, output_size)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.encoder_input_layer.bias.data.zero_()
        self.encoder_input_layer.weight.data.uniform_(-initrange, initrange)
    
    def generate_mask(self, dim1, dim2):
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
        
    def forward(self, src, src_mask=None):
        src = self.encoder_input_layer(src)
        src = self.positional(src)
        
        if src_mask is None:
            src_mask = self.generate_mask(self.window_len, self.window_len).to(src.device)
        
        encoder_output = self.encoder(src, src_mask)
        output = self.output_layer(encoder_output)
        
        return output


class StockTransformer(nn.Module):
    def __init__(self, window_len, target_len, num_encoder_layers, num_decoder_layers, input_size, output_size, d_model, num_heads,
                 feedforward_dim, dropout=0.1, positional_encoding="sinusoidal"):
        super(StockTransformer, self).__init__()
        
        if positional_encoding == "sinusoidal":
            self.positional_encoder = SinusoidalPositionalEncoder(seq_len=window_len, d_model=d_model, dropout=dropout)
            self.positional_decoder = SinusoidalPositionalEncoder(seq_len=target_len, d_model=d_model, dropout=dropout)
        elif positional_encoding == "learnable":
            self.positional_encoder = LearnablePositionalEncoder(seq_len=window_len, d_model=d_model, dropout=dropout)
            self.positional_decoder = LearnablePositionalEncoder(seq_len=target_len, d_model=d_model, dropout=dropout)
        else:
            raise Exception("Positional encoding type not recognized: use 'sinusoidal' or 'learnable'.")
        self.positional_encoding = positional_encoding
        
        self.window_len = window_len
        self.target_len = target_len
        
        self.encoder_input_layer = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=feedforward_dim, dropout=dropout,
                                                  batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)
        
        self.decoder_input_layer = nn.Linear(input_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=feedforward_dim, dropout=dropout,
                                                  batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers)
        
        self.output_layer = nn.Linear(d_model, output_size)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.encoder_input_layer.weight.data.uniform_(-initrange, initrange)
        self.decoder_input_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
    
    def generate_mask(self, dim1, dim2):
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
    
    def forward(self, src, trg, memory_mask=None, trg_mask=None):
        src = self.encoder_input_layer(src)
        src = self.positional_encoder(src)
        encoder_output = self.encoder(src)
        
        trg = self.decoder_input_layer(trg)
        trg = self.positional_decoder(trg)
        
        if memory_mask is None:
            memory_mask = self.generate_mask(self.target_len, self.window_len).to(src.device)
        if trg_mask is None:
            trg_mask = self.generate_mask(self.target_len, self.target_len).to(src.device)
        
        decoder_output = self.decoder(trg, encoder_output, trg_mask, memory_mask)
        
        output = self.output_layer(decoder_output)
        
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
        output = self.net(output[:, -1:, :])
        return output
