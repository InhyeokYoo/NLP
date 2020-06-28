import torch
import torch.nn as nn 
import torch.optim as optim
import random

# It is more oop friendly when Enc, Dec, Attention are implemented separably.
class Encoder(nn.Module):
    def __init__(self, rnn_type: str, vocab_size: int, emb_dim: int, hid_dim: int, dropout: float, n_layers: int=1, bidirectional: bool=True):
        super(Encoder, self).__init__()

        assert rnn_type in ['LSTM', 'RNN', 'GRU'], 'RNN type is not supported. Please select one of [GRU, RNN, LST]'

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.upper()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.num_dir = 2 if bidirectional else 1

        if rnn_type == "LSTM":
            self.encoder = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers, bidirectional=self.bidirectional, dropout=dropout)
        elif rnn_type == 'RNN':
            self.encoder = nn.RNN(emb_dim, hid_dim, num_layers=self.n_layers, bidirectional=self.bidirectional, dropout=dropout)
        elif rnn_type == 'GRU':
            self.encoder = nn.GRU(emb_dim, hid_dim, num_layers=self.n_layers, bidirectional=self.bidirectional, dropout=dropout)

        # for feeding initial hidden state to the decoder
        self.fc_h = nn.Linear(self.num_dir * hid_dim, hid_dim)
        if self.rnn_type == "LSTM":
            self.fc_c = nn.Linear(self.num_dir * hid_dim, hid_dim)

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
            c = c.view(self.n_layers, self.num_dir, batch_size, self.hid_dim) # [n_layers, num_dir, Batch, enc_hid]
            
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
        
        h = h.view(self.n_layers, self.num_dir, batch_size, self.hid_dim) # [n_layers, num_dir, Batch, enc_hid]
        
        if self.bidirectional:
            h = torch.cat([h[:, -1, :, :], h[:, 0, :, :]], dim=2) # [n_layers, Batch, 2 * enc_hid]
            h = h.permute(1, 0, 2) # [Batch, n_layers, 2 * enc_hid]
            h = torch.tanh(self.fc_h(h)) # [Batch, n_layers, dec_hid]

            H = H[:, :, :self.hid_dim] + H[:, :, self.hid_dim:] # [n_layers, Batch, enc_hid]
        else:
            h = h.squeeze(1) # [n_layers, Batch, 2 * enc_hid]
            h = h.permute(1, 0, 2) # [Batch, n_layers, 2 * enc_hid]
            h = torch.tanh(self.fc_h(h)) # [Batch, n_layers, dec_hid]
        
        h = h.permute(1, 0, 2)
        # H: [Seq_len, Batch, num_dir * hid]
        # h: [n_layers, Batch, dec_hid]
        if self.rnn_type == 'LSTM':
            # c: [n_layers, Batch, dec_hid]
            return H.contiguous(), (h.contiguous(), c.contiguous())
        else:
            return H.contiguous(), h.contiguous()

class LuongDecoder(nn.Module):
    def __init__(self, rnn_type: str, attn_type: str, vocab_size: int, emb_dim: int, hid_dim: int, attention: nn.Module, dropout: float,
                num_dir: int, device: str, n_layers: int=1, D: int=10, Pt: str='monotonic'):
        super(LuongDecoder, self).__init__()
        self.rnn_type = rnn_type.upper()
        self.attn_type = attn_type.lower()
        self.Pt = Pt.lower()

        assert self.rnn_type in ['LSTM', 'RNN', 'GRU'], 'RNN type is not supported. Please select one of [GRU, RNN, LST]'
        assert self.attn_type in ['global', 'local'], 'attention type is not supported. Please select one of [global, local]'
        assert self.Pt in ['monotonic', 'predictive'], 'Pt is not supported. Please select one of [monotonic, predictive]'
        
        self.D = D
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout)
        
        self.vocab_size = vocab_size
        self.num_dir = num_dir
        self.n_layers = n_layers
        self.attention = attention

        self.embedding = nn.Embedding(self.vocab_size, emb_dim)

        if self.rnn_type == "LSTM":
            self.decoder = nn.LSTM(self.hid_dim + emb_dim, hid_dim, dropout=dropout, num_layers=self.n_layers)
        elif self.rnn_type == 'RNN':
            self.decoder = nn.RNN(self.hid_dim + emb_dim, hid_dim, dropout=dropout, num_layers=self.n_layers)
        elif self.rnn_type == 'GRU':
            self.decoder = nn.GRU(self.hid_dim + emb_dim, hid_dim, dropout=dropout, num_layers=self.n_layers)

        if self.Pt == 'predictive':
            self.attn_dim = hid_dim // 8
            self.local_v = nn.Parameter(torch.randn(self.attn_dim, 1))
            self.local_w = nn.Linear(hid_dim, self.attn_dim)
        
        self.out_layer = nn.Linear(self.hid_dim, self.vocab_size)

    def forward(self, t: int, trg: torch.Tensor, prev_s: torch.Tensor, H: torch.Tensor):
        # trg(y_i): [Seq_len(=1), Batch_size]
        # prev_s: (h, c)
            # h: [n_layers, Batch, dec_hid]
            # c: [n_layers, Batch, dec_hid]
        # H: [Seq_len(T_x), Batch, enc_hid]
        batch_size = H.size(1)
        trg = trg.unsqueeze(0)
        self.t = t
        
        emb = self.embedding(trg) # -> emb: [Seq_len(=1), Batch_size, emb_dim]
        emb = self.dropout(emb)
        
        # Input feeding
        if self.rnn_type == 'LSTM':
            prev_c = prev_s[1]
            prev_s = prev_s[0] # [n_layers, Batch, dec_hid]

            temp = prev_s[-1].unsqueeze(0)
            input_ = torch.cat([temp, emb], dim=2) # [1, Batch_size, dec_hid + emb_dim]
            dec_output, (s, c) = self.decoder(input_, (prev_s, prev_c))
        else:
            temp = prev_s[-1].unsqueeze(0)
            input_ = torch.cat([temp, emb], dim=2) # [1, Batch_size, dec_hid + emb_dim]
            dec_output, s = self.decoder(input_, prev_s) 

        context = self._get_attn_weight(H, s[-1]) # [batch_size, 1, enc_hid]

        # then, concat [y_{i-1}, c_i] and feed to the decoder rnn
        context = context.permute(1, 0, 2) # [1, batch_size, enc_hid]

        # dec_output: [1, Batch, dec_hid]
        # s: [num_layer, Batch, dec_hid]
        # c: [num_layer, Batch, dec_hid]
        # Use only last layer's hidden state in decoder
        output = self.out_layer(s[-1, :, :].unsqueeze(0).permute(1, 0, 2)) # [Batch, 1, vocab_size]

        if self.rnn_type == 'LSTM':
            return output.squeeze(1), (s, c)
        else:
            return output.squeeze(1), s

    def _get_attn_weight(self, H, s):
        # s: [Batch, dec_hid]
        # H: [Seq_len(T_x), Batch, enc_hid]

        align_score  = self.attention(H, s) # [batch_size, 1, seq_len]
        batch_size = align_score.size(0)

        if self.attn_type == 'global':
            align_score = torch.softmax(align_score, dim=2) # [batch_size, 1, seq_len]
            context = torch.bmm(align_score, H.permute(1, 0, 2)) # [batch_size, 1, enc_hid]
        else:
            Tx = align_score.size(2)

            if self.Pt == 'monotonic':
                Pt = self.t # integer
                start = max(0, Pt - self.D)
                end = Pt + self.D + 1
                align_score = align_score[:, :, start:end] # [batch_size, 1, 2D + 1]
                H = H.permute(1, 0, 2)[:, start:end, :] # [batch_size, 1, 2D + 1]

                local = torch.tensor([x for x in range(1, align_score.size(2) + 1)]).to(device) # [2D + 1]
                local = torch.exp(- ((local -   Pt) ** 2)/((self.D ** 2) / 2)).view(1, 1, -1) # [2D + 1]
                local = local.repeat(batch_size, 1, 1) # [B, 1, 2D + 1]
                align_score = align_score * local  # [B, 1, 2D + 1]
                
                context = torch.bmm(align_score, H) # # [batch_size, 1, 2D + 1]
                
            elif self.Pt == 'predictive':
                # Pt - D and Pt + D are different for each batch
                temp = self.local_w(s.unsqueeze(1)) # [Batch, 1, attn_dim]
                Pt = Tx * torch.sigmoid(temp.bmm(self.local_v.repeat(batch_size, 1, 1))).view(-1, 1) # [Batch, 1]
                
                # Start/end represent Pt-D/Pt+D repectively
                start = torch.cat([Pt, torch.zeros_like(Pt)], dim=1)
                start = start.max(1)[0].long() # [batch_size]
                end =  start + self.D + 1 # [batch_size]
                
                # align_score = torch.cat([align_score[i, :, start[i]:end[i]] for i in range(batch_size)], dim=0).unsqueeze(1) # [batch_size, 1, 2D + 1]
                # list of batch_size len
                align_score = [align_score[i, :, start[i]:end[i]] for i in range(batch_size)] # [1, 2D + 1] for an element
                H = H.permute(1, 0, 2)
                # H = torch.cat([H[i, start[i]:end[i], :] for i in range(batch_size)], dim=0).view(batch_size, -1, self.hid_dim)
                # list of batch_size len
                H = [H[i, start[i]:end[i], :] for i in range(batch_size)] # [2D + 1, hid_dim] for an element
                
                # for j Pt - D, Pt + D
                end_batch = (start + torch.tensor([H[i].size(0) for i in range(len(H))]).to(device)).view(-1, 1) # [batch, 1]
                end = torch.cat([end.view(-1, 1), end_batch], dim=1) # [batch, 2]
                end = end.min(1)[0] # [batch]
                
                context = torch.zeros(batch_size, 1, self.hid_dim).to(device)

                for i in range(batch_size):
                    local = torch.tensor([j for j in range(start[i], end[i])]).to(device) # [2D + 1]: for j
                    batch_Pt = Pt[i]
                    local = torch.exp(-1 * ((local-batch_Pt) ** 2)/((self.D ** 2) / 2)) # [2D + 1]
                    score = align_score[i] * local  # [1, 2D + 1]
                    context[i] = score.mm(H[i])
        return context

class Attention(nn.Module):
    def __init__(self, rnn_type: str, align_fn: str, hid_dim: int, num_dir: int):
        super(Attention, self).__init__()
        self.align_fn = align_fn.lower()
        self.rnn_type = rnn_type.upper()

        assert self.align_fn in ['concat', 'general', 'dot'], 'Alignment model is not supported. Please select one of [concat, general, dot]'

        self.num_dir = num_dir
        self.hid_dim = hid_dim

        if self.align_fn == 'concat':
            self.attn_dim = hid_dim // 8
            self.v = nn.Parameter(torch.rand(1, self.attn_dim)) # [1, attn_dim] 
            self.attn = nn.Linear(self.hid_dim + self.hid_dim, self.attn_dim)
        elif self.align_fn == 'general':
            self.attn = nn.Linear(self.hid_dim, self.hid_dim)


    def forward(self, H, prev_s):
        # H: [Seq_len, Batch, enc_hid]
        # prev_s: (h, c)
            # h: [Batch, dec_hid]
            # c: [Batch, dec_hid]

        batch_size = H.size(1)

        if self.align_fn == 'concat':
            # prev_s: [Batch, dec_hid]
            # vT tanh(Ws + Uh)
            prev_s = prev_s.unsqueeze(1) # [Batch, 1, dec_hid]
            prev_s = prev_s.repeat(1, H.size(0), 1) # [Batch, seq_len, dec_hid]
            H = H.permute(1, 0, 2) # [Batch, Seq_len, enc_hid]
            a = torch.cat([prev_s, H], dim=2) # [batch_size, seq_len, dec_hid + enc_hid]
            a = torch.tanh(self.attn(a)) # [batch_size, seq_len, attn_dim]
            a = a.permute(0, 2, 1)  # [batch_size, attn_dim, seq_len(=1)]
            v = self.v.unsqueeze(0).expand(batch_size, -1, -1) # [batch_size, 1, attn_dim]
            align_score = torch.bmm(v, a) # [batch_size, 1, seq_len]
            
            return align_score

        elif self.align_fn == 'general':
            H = H.permute(1, 0, 2) # [Batch, Seq_len, enc_hid]
            a = self.attn(H) # [Batch, Seq_len, enc_hid]
            
            align_score = torch.bmm(a, prev_s.unsqueeze(2)) # [batch_size, seq_len, 1]
            align_score = align_score.permute(0, 2, 1) # [batch_size, 1, seq_len]

            return align_score

        elif self.align_fn == 'dot':
            H = H.permute(1, 2, 0) # [Batch, enc_hid, Seq_len]
            align_score = torch.bmm(prev_s.unsqueeze(1), H) # [batch_size, 1, seq_len]

            return align_score
            
class Seq2Seq(nn.Module):
    def __init__(self, rnn_type: str, encoder: nn.Module, decoder: nn.Module, trg_vocab_size: int, device: str, num_dir: int, 
                teacher_forcing: float=0.5, beam_search: int=1):
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
                y, (s, c) = self.decoder(t, y, (s, c), H)
            else:
                y, s = self.decoder(t, y, s, H)      
            outputs[t] = y
            is_teacher_force = random.random() < self.teacher_forcing
            topk = y.max(1)[1] # get idx of the next word
            y = targets[t] if is_teacher_force else topk

        return outputs