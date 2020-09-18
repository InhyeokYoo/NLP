from typing import Iterable, Optional, List
import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool1d

class ELMO(nn.Module):
    def __init__(self, vocab_size, emb_dim: int, hid_dim: int, n_layers: int, dropout: Optional[float]):
        super(ELMO, self).__init__()
        self.embedding = CharCNN(vocab_size, emb_dim)
        self.biLM = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        

    def forward(self, x):
        '''
        param:

        dim:
            x: [batch, seq_len]
        '''
        pass


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, kernel_sizes: List[int], seq_len: int):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.seq_len = seq_len
        self.kernel_sizes = kernel_sizes
        self.kernels = [nn.Conv1d(in_channels=emb_dim, out_channels=1, kernel_size=kernel_size)
                       for kernel_size in kernel_sizes] # [B, emb_dim, seq_len] -> # [B, 1, seq_len - kernel_size + 1]
        # TODO: Vocab 만들기: SOW, EOW, PAD index 포함
        self.vocab = dict()

    def forward(self, word: Iterable):
        '''
        param:
            word: a word, NOT A TENSOR!
        '''
        batch_size = len(word)
        word_matrix = torch.zeros(batch_size, self.emb_dim, self.seq_len) # [B, emb_dim, seq_len]

        for i in range(self.seq_len):
            # TODO: character_vocab.py 완성
            char_idx = [self.vocab[char[i]] for char in word]  # get char index
            x = self.embedding(char_idx) # [B, ]
            word_matrix[:, i] = x

        # torch.max -> MaxBackward1
        y = torch.zeros(batch_size, self.kernel_sizes)

        for i in range(self.kernel_sizes):
            temp = self.kernels[i](word_matrix).squeeze(1) # [B, l+2-w+1]
            y[:, i] = torch.max(temp, dim=1)
        

    def char_pad(self):
        pass

class HighwayNetwork(nn.Module):
    def __init__(self):
        pass