import torch
import torch.nn as nn 
import torch.optim as optim
import random

# It is more oop friendly when Enc, Dec, Attention are implemented separably.
class Encoder(nn.Module):
    def __init__(self, rnn_type: str, vocab_size: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float, n_layers: int=1, bidirectional: bool=True):
        super(Encoder, self).__init__()

        assert rnn_type in ['LSTM', 'RNN', 'GRU'], 'RNN type is not supported. Please select one of [GRU, RNN, LST]'

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.upper()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.num_dir = 2 if bidirectional else 1

        if rnn_type == "LSTM":
            self.encoder = nn.LSTM(emb_dim, enc_hid_dim, num_layers=self.n_layers, bidirectional=self.bidirectional, dropout=dropout)
        elif rnn_type == 'RNN':
            self.encoder = nn.RNN(emb_dim, enc_hid_dim, num_layers=self.n_layers, bidirectional=self.bidirectional, dropout=dropout)
        elif rnn_type == 'GRU':
            self.encoder = nn.GRU(emb_dim, enc_hid_dim, num_layers=self.n_layers, bidirectional=self.bidirectional, dropout=dropout)

        # for feeding initial hidden state to the decoder
        self.fc_h = nn.Linear(self.num_dir * enc_hid_dim, dec_hid_dim)
        if self.rnn_type == "LSTM":
            self.fc_c = nn.Linear(self.num_dir * enc_hid_dim, dec_hid_dim)

    def forward(self, src: torch.Tensor, hidden=None):
        # src: [seq_len, Batch_size]
        batch_size = src.size(1)
        seq_len = src.size(0)

        emb = self.embedding(src) # -> emb: [seq_len, Batch_size, emb_dim]
        emb = self.dropout(emb)
        
        # if initial hidden state were not fed, zero vectors are provided automatcially
        # Use only last layer's hidden state of stacked RNN
        if self.rnn_type == 'LSTM':
            H, (h, c) = self.encoder(emb, hidden) # don't need cell state
            # c: [num_dir * n_layers, Batch, enc_hid]
            c = c.view(self.n_layers, self.num_dir, batch_size, self.enc_hid_dim) # [n_layers, num_dir, Batch, enc_hid]
            
            if self.bidirectional:
                c = torch.cat([c[:, -1, :, :], c[:, 0, :, :]], dim=2) # [n_layers, Batch, 2 * enc_hid]
                c = c.permute(1, 0, 2) # [Batch, n_layers, 2 * enc_hid]
                c = torch.tanh(self.fc_c(c)) # [Batch, n_layers, dec_hid]
            else:
                c = c.squeeze(1) # [n_layers, Batch, 2 * enc_hid]
                c = c.permute(1, 0, 2) # [Batch, n_layers, 2 * enc_hid]
                c = torch.tanh(self.fc_c(c)) # [Batch, n_layers, dec_hid]
            # c: [Batch, n_layers, dec_hid]
            c = c.permute(1, 0, 2)
            # c: [n_layers, Batch, dec_hid]
        else: 
            H, h = self.encoder(emb, hidden)
        # H: [Seq_len, Batch, num_dir * hid]
        # h: [num_dir * n_layers, Batch, enc_hid]
        
        h = h.view(self.n_layers, self.num_dir, batch_size, self.enc_hid_dim) # [n_layers, num_dir, Batch, enc_hid]
        
        if self.bidirectional:
            h = torch.cat([h[:, -1, :, :], h[:, 0, :, :]], dim=2) # [n_layers, Batch, 2 * enc_hid]
            h = h.permute(1, 0, 2) # [Batch, n_layers, 2 * enc_hid]
            h = torch.tanh(self.fc_h(h)) # [Batch, n_layers, dec_hid]
        else:
            h = h.squeeze(1) # [n_layers, Batch, 2 * enc_hid]
            h = h.permute(1, 0, 2) # [Batch, n_layers, 2 * enc_hid]
            h = torch.tanh(self.fc_h(h)) # [Batch, n_layers, dec_hid]
        
        h = h.permute(1, 0, 2)
        # H: [Seq_len, Batch, num_dir * hid]
        # h: [n_layers, Batch, dec_hid]
        if self.rnn_type == 'LSTM':
            # c: [n_layers, Batch, dec_hid]
            return H, (h, c)
        else:
            return H, h

class BahdanauDecoder(nn.Module):
    def __init__(self, rnn_type: str, vocab_size: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, attention: nn.Module, dropout: float,
                num_dir: int, n_layers: int=1):
        super(BahdanauDecoder, self).__init__()

        assert rnn_type in ['LSTM', 'RNN', 'GRU'], 'RNN type is not supported. Please select one of [GRU, RNN, LST]'

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = nn.Dropout(dropout)
        self.rnn_type = rnn_type.upper()
        self.vocab_size = vocab_size
        self.num_dir = num_dir
        self.n_layers = n_layers
        self.attention = attention

        self.embedding = nn.Embedding(self.vocab_size, emb_dim)

        if self.rnn_type == "LSTM":
            self.decoder = nn.LSTM((self.num_dir * self.enc_hid_dim) + emb_dim, dec_hid_dim, dropout=dropout, num_layers=self.n_layers)
        elif self.rnn_type == 'RNN':
            self.decoder = nn.RNN((self.num_dir * self.enc_hid_dim) + emb_dim, dec_hid_dim, dropout=dropout, num_layers=self.n_layers)
        elif self.rnn_type == 'GRU':
            self.decoder = nn.GRU((self.num_dir * self.enc_hid_dim) + emb_dim, dec_hid_dim, dropout=dropout, num_layers=self.n_layers)

        self.out_layer = nn.Linear(self.dec_hid_dim, self.vocab_size)

    def forward(self, trg: torch.Tensor, prev_s: torch.Tensor, H: torch.Tensor):
        # trg(y_i): [Seq_len(=1), Batch_size]
        # H: [Seq_len, Batch, num_dir * hid]
        # prev_s: (h, c)
            # h: [n_layers, Batch, dec_hid]
            # c: [n_layers, Batch, dec_hid]
        batch_size = H.size(1)
        trg = trg.unsqueeze(0)
        
        emb = self.embedding(trg) # -> emb: [Seq_len(=1), Batch_size, hid_dim]
        emb = self.dropout(emb)
        
        context = self._get_attn_weight(H, prev_s) # -> [batch_size, 1, num_dir * enc_hid]

        # then, concat [y_{i-1}, c_i] and feed to the decoder rnn
        context = context.permute(1, 0, 2) # [1, batch_size, num_dir * enc_hid]

        input_ = torch.cat([emb, context], dim=2) # [1, batch_size, emd_dim + (num_dir * enc_hid)]
        if self.rnn_type == 'LSTM':
            dec_output, (s, c) = self.decoder(input_, (prev_s[0].contiguous(), prev_s[1].contiguous()))

        else:
            dec_output, s = self.decoder(input_, prev_s.contiguous())

        output = self.out_layer(s[-1, :, :].unsqueeze(0).permute(1, 0, 2)) # [Batch, 1, vocab_size]

        if self.rnn_type == 'LSTM':
            return output.squeeze(1), (s, c)
        else:
            return output.squeeze(1), s

    def _get_attn_weight(self, H, s):
        # prev_s: (h, c)
            # h: [n_layers, Batch, dec_hid]
            # c: [n_layers, Batch, dec_hid]
        align_score  = self.attention(H, s) # [batch_size, 1, seq_len]
        context = torch.bmm(align_score, H.permute(1, 0, 2)) # [batch_size, 1, num_dir * enc_hid]
        return context

class Attention(nn.Module):
    def __init__(self, rnn_type: str, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int, num_dir: int):
        super(Attention, self).__init__()
        self.rnn_type = rnn_type.upper()

        self.attn_dim = attn_dim
        self.num_dir = num_dir
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.v = nn.Parameter(torch.rand(1, self.attn_dim)) # [1, attn_dim] 
        self.attn = nn.Linear(self.num_dir * self.enc_hid_dim + self.dec_hid_dim, self.attn_dim)

    def forward(self, H, prev_s):
        # H: [Seq_len, Batch, num_dir * hid]
        # prev_s: (h, c)
            # h: [n_layers, Batch, dec_hid]
            # c: [n_layers, Batch, dec_hid]
        
        batch_size = H.size(1)
        if self.rnn_type == 'LSTM':
            prev_s = prev_s[0]
        
        # prev_s: [n_layers, Batch, dec_hid]
        # vT tanh(Ws + Uh)
        prev_s = prev_s[-1].unsqueeze(1) # [Batch, 1, dec_hid]
        prev_s = prev_s.repeat(1, H.size(0), 1) # [Batch, seq_len, dec_hid]
        H = H.permute(1, 0, 2) # [Batch, Seq_len, num_dir * enc_hid]
        a = torch.cat([prev_s, H], dim=2) # [batch_size, seq_len, dec_hid + (num_dir * enc_hid)]
        a = torch.tanh(self.attn(a)) # [batch_size, seq_len, attn_dim]
        a = a.permute(0, 2, 1)  # [batch_size, attn_dim, seq_len(=1)]
        v = self.v.unsqueeze(0).expand(batch_size, -1, -1) # [batch_size, 1, attn_dim]
        align_score = torch.bmm(v, a) # [batch_size, 1, seq_len]
        
        return torch.softmax(align_score, dim=2) # [batch_size, 1, seq_len]

class Seq2Seq(nn.Module):
    def __init__(self, rnn_type: str, encoder: nn.Module, decoder: nn.Module, trg_vocab_size: int, enc_hid_dim: int, dec_hid_dim: int,
                device: str, num_dir: int, teacher_forcing: float=0.5, 
                beam_search: int=1):
        super(Seq2Seq, self).__init__()

        self.beam_search = beam_search
        self.teacher_forcing = teacher_forcing
        self.device = device
        self.num_dir = num_dir
        self.trg_vocab_size = trg_vocab_size
        self.encoder = encoder
        self.decoder = decoder
        self.rnn_type = rnn_type.upper()

    def forward(self, sources, targets):
        batch_size = sources.size(1) 
        max_len = targets.size(0) # only search sentences of length up to

        outputs = torch.zeros(max_len, batch_size, self.trg_vocab_size).to(self.device)
        if self.rnn_type == 'LSTM':
            H, (h, c) = self.encoder(sources)
            # c: [Batch, dec_hid]
        else:
            H, h = self.encoder(sources)
            # h: [Batch, dec_hid]
        
        # H: [Seq_len, Batch, num_dir * enc_hid]

        s = h
        y = targets[0, :]

        for t in range(1, max_len):
            if self.rnn_type == 'LSTM':
                y, (s, c) = self.decoder(y, (s, c), H)
            else:
                y, s = self.decoder(y, s, H)      
            outputs[t] = y
            is_teacher_force = random.random() < self.teacher_forcing
            topk = y.max(1)[1] # get idx of the next word
            y = targets[t] if is_teacher_force else topk

        return outputs