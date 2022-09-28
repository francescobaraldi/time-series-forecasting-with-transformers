import torch
import torch.nn as nn
from positional_encoding import PositionalEncoderStd, Time2Vec


class Transformer(nn.Module):
    def __init__(self, seq_len, num_encoder, num_decoder, input_size, output_size, d_model, num_heads, feedforward_dim):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.positional = PositionalEncoderStd(d_model=d_model)
        self.encode_input_layer = nn.Linear(input_size, d_model)
        encode_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=feedforward_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encode_layer, num_encoder)
        # self.positional = nn.parameter.Parameter(torch.rand(d_model))
        self.decode_input_layer = nn.Linear(input_size, d_model)
        decode_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=feedforward_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decode_layer, num_layers=num_decoder)
        self.output_layer = nn.Linear(d_model, input_size)
        
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


class TransformerDecoder(nn.Module):
    def __init__(self, seq_len, num_layer, input_size, output_size, d_model, num_heads, feedforward_dim):
        super(TransformerDecoder, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        # self.positional = Time2Vec(input_size, d_model)
        self.encode_input_layer = nn.Linear(input_size, d_model)
        encode_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=feedforward_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encode_layer, num_layer)
        self.output_layer = nn.Linear(d_model, output_size)
        
    def forward(self, src, src_mask=None):
        src = self.encode_input_layer(src)
        # src_pos = self.positional(src)
        
        if src_mask is None:
            src_mask = self.generate_mask(self.seq_len, self.seq_len).to(src.device)
        
        encoder_output = self.encoder(src, src_mask)
        output = self.output_layer(encoder_output)
        
        return output
        
    
    def generate_mask(self, dim1, dim2):
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


class TransformerDecoder_v2(nn.Module):
    def __init__(self, seq_len, num_layer, input_size, output_size, num_heads, feedforward_dim):
        super(TransformerDecoder_v2, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        encode_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=feedforward_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encode_layer, num_layer)
        self.output_layer = nn.Linear(input_size, output_size)
        
    def forward(self, src, src_mask=None):
        if src_mask is None:
            src_mask = self.generate_mask(self.seq_len, self.seq_len).to(src.device)
        
        encoder_output = self.encoder(src, src_mask)
        output = self.output_layer(encoder_output)
        
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
