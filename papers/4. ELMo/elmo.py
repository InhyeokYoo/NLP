from typing import Optional, List
import torch
import torch.nn as nn

class ELMo(nn.Module):
    def __init__(self, vocab_size, emb_dim: int, hid_dim: int, kernel_sizes: List[int], seq_len: int,
                 n_layers: int, dropout: Optional[float]):
        super(ELMo, self).__init__()

        kernel_dim = sum([kernel_size * 25 for kernel_size in kernel_sizes])

        self.embedding = CharCNN(vocab_size, emb_dim, kernel_sizes, seq_len)
        self.bilms = [nn.LSTM(emb_dim, hid_dim, bidirectional=True, dropout=dropout) for _ in range(n_layers)]

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
        self.kernel_dim = sum([kernel_size * 25 for kernel_size in kernel_sizes]) # for the small model of the paper
        self.kernels = [nn.Conv1d(in_channels=emb_dim, out_channels=25*kernel_size, kernel_size=kernel_size)
                       for kernel_size in kernel_sizes] # [B, emb_dim, seq_len] -> # [B, 1, seq_len - kernel_size + 1]
        self.highway_net = HighwayNetwork(self.kernel_dim)
        
        # TODO: Vocab 만들기: SOW, EOW, PAD index 포함
        self.vocab = dict()

    def forward(self, word: torch.Tensor):
        '''
        The only important difference is that we use a larger number ofconvolutional features of 4096 to give enough capacity tothe model.
        param:
            word: a word
        dim:
            [B, emb_dim, seq_len]
        '''
        batch_size = word.size(0)
        y = torch.zeros(batch_size, self.kernel_dim)
 
        cnt = 0

        for kernel in self.kernels:
            temp = kernel(word)
            pooled = torch.max(temp, dim=2)[0]
            y[:, cnt:cnt+pooled.size(1)] = pooled
            cnt += pooled.size(1)
        '''
        # cat vs. fill empty tensor
        y = []
        for kernel in kernels:
            temp = kernel(a)
            y.append(torch.max(temp, dim=2)[0]) # max pooling
        y = torch.cat(y, dim=1)
        '''

        return self.highway_net(y)
    
    def char_pad(self):
        pass

class HighwayNetwork(nn.Module):
    def __init__(self, kernel_sizes: int):
        super(HighwayNetwork, self).__init__()
        self.linear = nn.Linear(kernel_sizes, kernel_sizes)
        self.t_gate = nn.Sequential(nn.Linear(kernel_sizes, kernel_sizes), nn.Sigmoid())

    def forward(self, y):
        t = self.t_gate(y)
        c = 1 - t
        return t * torch.relu(self.linear(y)) + c * y

    def _init_bias(self):
        '''
        Srivastava et al. (2015) recommend initializing b_T to a negative vlaue, 
        in order to militate the initial behavior towars carry. We initialized b_T to a
        small interval around -2.
        '''
        self.t_gate[0].bias.data.fill_(-2)