import torch
import torch.nn as nn
from typing import List

class CharEmbedding(nn.Module):
    r"""
    > The only important difference is that we use a larger number of convolutional features of 4096 to give enough capacity to the model (Jozefowicz et al. 2016).
    > The context insensitive type representation uses 2048 character n-gram convolutional filters followed by two highway layers
    and a linear projection down to a 512 representation (Peters et al. 2018).
    """
    def __init__(self, vocab_size: int, emb_dim: int, prj_dim: int, kernel_sizes: List[List[int]], char_len: int, device: str):
        super().__init__()
        self.device = device
        self.kernel_dim = sum([kernel_size  for num_features, kernel_size in kernel_sizes]) # same as the embedding dim
        self.charcnn = CharCNN(vocab_size, emb_dim, self.kernel_dim, kernel_sizes, char_len, device)
        self.highway_net = HighwayNetwork(self.kernel_dim)
        self.highway_net._init_bias()
        self.projection_layer = nn.Linear(self.kernel_dim, prj_dim)

    def forward(self, x):
        r"""
        Parameters:
            x: A sentence vector composed of characters of the sentence
        Dimensions:
            x: [Batch, Seq_len, Char_len]
        """
        batch_size, seq_len, _ = x.size()
        y = torch.zeros(batch_size, seq_len, self.kernel_dim).to(self.device) # [Batch, Seq_len, Kernel_dim]
        
        for i in range(seq_len):
            char_emb = self.charcnn(x[:, i, :]) # [Batch, Kernel_dim]
            highway_emb = self.highway_net(char_emb) # [Batch, 1, Kernel_dim]
            y[:, i, :] = highway_emb.squeeze(1)
        
        emb = self.projection_layer(y) # [Batch, Seq_len, Projection_layer (==Emb_dim==HId_dim)]
        return emb

class CharCNN(nn.Module):
    r"""
    An implementation of 'Character-Aware Neural Language Models' of Kim et al. (2015).
    """
    def __init__(self, vocab_size: int, char_emb_dim: int, word_emb_dim: int, kernel_sizes: List[List[int]], char_len: int, device: str):
        r"""
        Parameters:
            vocab_size: `int`. Vocabulary size of chracters used in the model
            emb_dim: `int`. Embedding size
            kernel_sizes: A `list` of `list`s of `int`s. The nested list indicates feature maps for the convolutions in the paper (i.e. [(kernel_size, # kernels), ...])
            char_len: 'int'. Character length. The rest place of a character is padded
        """
        super(CharCNN, self).__init__()
        self.device = device
        self.char_len = char_len
        self.word_emb_dim = word_emb_dim
        self.kernel_sizes = kernel_sizes

        self.embedding = nn.Embedding(vocab_size, char_emb_dim)
        self.kernels = nn.ModuleList([nn.Conv1d(in_channels=char_emb_dim, out_channels=num_features, kernel_size=kernel_size)
                                                        for kernel_size, num_features in kernel_sizes]) # [Batch, Emb_dim, char_len] -> # [Batch, 1, Char_len - Kernel_size + 1]

    def forward(self, word: torch.Tensor):
        r"""
        Parameters:
            word: an input tensor. 
        Dimensions:
            input:
                word: [Batch, Emb_dim, Seq_len]
            output:
                y:[Batch, Kernel_dim]
        """
        batch_size = word.size(0)
        y = torch.zeros(batch_size, self.word_emb_dim).to(self.device) # [Batch, Kernel_dim]
 
        cnt = 0 # index for y

        # filling an empty tensor is slightly faster than `torch.cat`
        for kernel in self.kernels:
            emb = self.embedding(word) # [Batch, Kernel_dim, Emb_dim]
            emb = emb.permute(0, 2, 1) # [Batch, Emb_dim, Kernel_dim]
            temp = kernel(emb) # [Batch, kernel_sizes[1], Char_len - w + 1]
            pooled = torch.max(temp, dim=2)[0] # [Batch, kernel_sizes[1]]
            y[:, cnt:cnt+pooled.size(1)] = pooled
            cnt += pooled.size(1)

        return y # [Batch, Kernel_dim]

class HighwayNetwork(nn.Module):
    def __init__(self, kernel_sizes: int):
        super(HighwayNetwork, self).__init__()
        self.h_gate = nn.Linear(kernel_sizes, kernel_sizes)
        self.t_gate = nn.Sequential(nn.Linear(kernel_sizes, kernel_sizes), nn.Sigmoid())
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Dimensions
            x: [Batch, Kernel_dim]
        """ 
        x = x.unsqueeze(1) # [Batch, 1, Kernel_dim]
        h = self.relu(self.h_gate(x)) # [Batch, 1, Kernel_dim]
        t = self.t_gate(x) # [Batch, 1, Kernel_dim]
        c = 1 - t
        return t * h + c * x # [Batch, 1, Kernel_dim]

    def _init_bias(self):
        r"""
        > Srivastava et al. (2015) recommend initializing b_T to a negative vlaue, in order to militate the initial behavior towars carry. 
        We initialized b_T to a small interval around -2.
        """
        self.t_gate[0].bias.data.fill_(-2)