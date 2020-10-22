import torch
import torch.nn as nn
from transformer import Decoders

# voc_size:int, seq_len: int, d_model: int, d_ff:int, num_decoder: int, num_heads: int, dropout: float)

class LanguageModeling(nn.Module):
    def __init__(self, voc_size:int, seq_len: int, d_model: int, d_ff:int, num_decoder: int, num_heads: int, dropout: float) -> None:
        super(LanguageModeling, self).__init__
        self.decoders = Decoders(voc_size, seq_len, d_model, d_ff, num_decoder, num_heads, dropout)
        self.softmax = nn.softmax()
        
    def forward(self, x):
        emb = self.decoders(x)
        result = self.softmax(emb)

        return result