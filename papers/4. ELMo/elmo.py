from typing import Optional
import torch
import torch.nn as nn

class ELMO(nn.Module):
    def __init__(self, emb_dim: int, hid_dim: int, n_layers: int, dropout: Optional[float]):
        super(ELMO, self).__init__()
        self.embedding = CharCNN(emb_dim)
        self.biLM = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)

    def forward(self, x):
        token_representation = self.embedding(x)

class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x):
        '''
        param:
            x: characters
        dim:
            x: [batch, 
        '''
        pass

