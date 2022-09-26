import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, dropout: float=0.1, max_seq_len: int=5000, d_model: int=512, batch_first: bool=True):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        # adapted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)
            
            pe[0, :, 0::2] = torch.sin(position * div_term)
            
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)
        
            pe[:, 0, 0::2] = torch.sin(position * div_term)
        
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, seq_len, num_encoder, num_decoder, input_size, output_size, d_model, num_heads, feedforward_dim):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.positional = PositionalEncoder(d_model=d_model)
        self.encode_input_layer = nn.Linear(input_size, d_model)
        encode_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=feedforward_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encode_layer, num_encoder)
        # self.positional = nn.parameter.Parameter(torch.rand(d_model))
        self.decode_input_layer = nn.Linear(input_size, d_model)
        decode_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=feedforward_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decode_layer, num_layers=num_decoder)
        self.output_layer = nn.Linear(d_model, output_size)
        
    def forward(self, src, trg, src_mask=None, trg_mask=None):
        src = self.encode_input_layer(src)
        src_pos = self.positional(src)
        encoder_output = self.encoder(src_pos)
        
        trg = self.decode_input_layer(trg)
        
        if src_mask is None:
            src_mask = self.generate_mask(self.output_size, self.seq_len).to(src.device)
        if trg_mask is None:
            trg_mask = self.generate_mask(self.output_size, self.output_size).to(src.device)
        
        decoder_output = self.decoder(tgt=trg, memory=encoder_output, tgt_mask=trg_mask, memory_mask=src_mask)
        
        output = self.output_layer(decoder_output)
        
        return output
        
    
    def generate_mask(self, dim1, dim2):
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
           

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
