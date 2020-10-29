from typing import List, Tuple
import torch
import torch.nn as nn
from char_cnn import CharEmbedding

class ELMo(nn.Module):
    def __init__(self, vocab_size, output_dim: int, emb_dim: int, hid_dim: int, prj_dim, kernel_sizes: List[Tuple[int]], 
                seq_len: int, n_layers: int=2, dropout: float=0.):
        r"""
        Parametrs:
            vocab_size: Character vocabulary size
            output_dim: Word vocabulary size
            emb_dim: Embedding dimension of chracter tokens
            hid_dim: hidden dimension for bi-directional language model
            kernel_sizes: `list`. Kernel_sizes for the convolution operations.
            seq_len: character sequence len
            n_layers: the number of layers of LSTM. default 2
            dropout: dropout of LSTM
        """
        super(ELMo, self).__init__()

        self.embedding = CharEmbedding(vocab_size, emb_dim, prj_dim, kernel_sizes, seq_len)
        self.bilms = BidirectionalLanguageModel(hid_dim, hid_dim, n_layers, dropout)

        self.predict = nn.Linear(hid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor):
        r"""
        Parameters:
            x: Sentence
        Dimensions:
            x: [batch, seq_len]
        """
        emb = self.embedding(x) # [Batch, Seq_len, Projection_layer (==Emb_dim==HId_dim)]
        _, last_output = self.bilms(emb) # [Batch, Seq_len, Hidden_size]
        predict = self.predict(last_output) # [Batch, Seq_len, VOCAB_SIZE]
        y = self.softmax(predict)

        return y # only use the output of the last LSTM of the biLM on training step

    def get_embed_layers(self, x: torch.Tensor) -> List:
        r"""
        Same as the forward, but return embedding all of layers

        Parameters:
            x: Sentence. The sentence is composed by characeters. 
        Dimensions:
            x: [batch, seq_len]
        """
        emb = self.embedding(x) # [Batch, Seq_len, Projection_layer (==Emb_dim==HId_dim)]
        fisrt_output, last_output = self.bilms(emb) # [Batch, Seq_len, Hidden_size]

        return emb, (fisrt_output, last_output)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # > LSTM forget gate were iniailized to 1.0 (Jozefowicz et al. 2016)
        for lstm in self.bilms.lstms:
            for names in lstm._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(lstm, name)
                    n = bias.size(0)
                    start, end = n//4, n//2
                    bias.data[start:end].fill_(1.)

class BidirectionalLanguageModel(nn.Module):
    def __init__(self, emb_dim: int, hid_dim: int, prj_emb: int, dropout: float=0.) -> None:
        r"""
        > We use dropout before and after evert LSTM layer
        """
        super(BidirectionalLanguageModel, self).__init__()
        self.lstms = nn.ModuleList([nn.LSTM(emb_dim, hid_dim, bidirectional=True, dropout=dropout, batch_first=True), 
                      nn.LSTM(prj_emb, hid_dim, bidirectional=True, dropout=dropout, batch_first=True)])
        self.projection_layer = nn.Linear(2*hid_dim, prj_emb)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor]=None):
        r"""
        Parameters:
            x: A sentence tensor that embeded
            hidden: tuple of hidden and cell. The initial hidden and cell state of the LSTM
        Dimensions:
            x: [Batch, Seq_len, Emb_size]
            hidden: [num_layers * num_directions, batch, hidden_size], [num_layers * num_directions, batch, hidden_size]
        """
        # > "...add a residual connection between LSTM layers"
        first_output, (hidden, cell) = self.lstms[0](x, hidden) # [Batch, Seq_len, # directions * Hidden_size]
        # TODO: [Batch, Seq_len, # directions * Hidden_size]를 그냥 넣는지, 
        #       [Batch, Seq_len, # directions, Hidden_size]로 바꾼 후(nn.projection) 더해서 넣는지 확인이 필요
        projected = self.projection_layer(first_output) # [Batch, Seq_len, Projection_size]
        second_output, (hidden, cell) = self.lstms[1](projected, (hidden, cell)) # [Batch, Seq_len, # directions * Hidden_size]

        second_output = second_output.view(second_output.size(0), second_output.size(1), 2, -1) # [Batch, Seq_len, # directions, Hidden_size]
        second_output = second_output[:, :, 0, :] + second_output[:, :, 1, :] # [Batch, Seq_len, Hidden_size]
        return first_output, second_output   # [Batch, Seq_len, Hidden_size]