from typing import Optional
import torch
import torch.nn as nn

class MaskedLanguageModeling(nn.Module):
    def __init__(self, bert: nn.Module, voc_size:int=30000):
        super(MaskedLanguageModeling, self).__init__()
        self.bert = bert
        d_model = bert.emb.tok_emb.weight.size(1)
        self.linear = nn.Linear(d_model, voc_size)

    def forward(self, input: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        '''
        param:
            input: a batch of sequences of words
            seg: Segmentation embedding for input tokens
        dim:
            input:
                input: [B, S]
                seg: [B, S]
            output:
                result: [B, S, V]
        '''
        output = self.bert(input, seg) # [B, S, D_model]
        output = self.linear(output) # [B, S, voc_size]

        return output # [B, S, voc_size]

class NextSentencePrediction(nn.Module):
    def __init__(self, bert: nn.Module):
        super(NextSentencePrediction, self).__init__()
        self.bert = bert
        d_model = bert.emb.tok_emb.weight.size(1)
        self.linear = nn.Linear(d_model, 2)

    def forward(self, input: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        '''
        param:
            input: a batch of sequences of words
            seg: Segmentation embedding for input tokens
        dim:
            input:
                input: [B, S]
                seg: [B, S]
            output:
                result: [B, S, V]
        '''
        output = self.bert(input, seg) # [B, S, D_model]
        output = self.linear(output) # [B, S, 2]

        return output[:, 0, :] # [B, 2]

class BertModel(nn.Module):
    def __init__(self, voc_size:int=30000, seq_len: int=512, d_model: int=768, d_ff:int=3072, pad_idx: int=1,
                num_encoder: int=12, num_heads: int=12, dropout: float=0.1):
        super(BertModel, self).__init__()
        self.pad_idx = pad_idx
        self.emb = BERTEmbedding(seq_len, voc_size, d_model, dropout)
        self.encoders = Encoders(seq_len, d_model, d_ff, num_encoder, num_heads, dropout)

    def forward(self, input: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        '''
        param:
            input: a batch of sequences of words
        dim:
            input:
                input: [B, S]
            output:
                result: [B, S, V]
        '''
        pad_mask = get_attn_pad_mask(input, input, self.pad_idx)

        emb = self.emb(input, seg) # [B, S, D_model]
        output = self.encoders(emb, pad_mask) # [B, S, D_model]

        return output # [B, S, D_model]

class Encoders(nn.Module):
    def __init__(self, seq_len: int, d_model: int, d_ff:int, num_encoder: int, num_heads: int, dropout: float):
        super(Encoders, self).__init__()
        self.models = nn.ModuleList([Encoder(d_model, d_ff, seq_len, num_heads, dropout=dropout) for i in range(num_encoder)])
        
    def forward(self, tokens, enc_mask):
        # input: [B, S, D_model]
        for model in self.models:
            tokens = model(tokens, enc_mask)

        return tokens

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: int=0.1):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(seq_len, d_model)
        
    def forward(self, x: torch.Tensor):
        # x: [B, S]. x is tokens
        pos = torch.arange(self.seq_len, dtype=torch.long, device=x.device) # [S]
        pos = pos.unsqueeze(0).expand(x.size()) # [1, S] -> [B, S]
        pos_emb = self.emb(pos)
        return self.dropout(pos_emb) # [B, S, D_model]

class BERTEmbedding(nn.Module):
    """
    Embeddings for BERT.
    It includes segmentation embedding, token embedding and positional embedding.
    I add dropout for every embedding layer just like the original transformer.
    """
    def __init__(self, seq_len: int=512, voc_size: int=30000, d_model: int=768, dropout: float=0.1) -> None:
        super(BERTEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(num_embeddings=voc_size, embedding_dim=d_model)
        self.tok_dropout = nn.Dropout(dropout)
        self.seg_emb = nn.Embedding(2, d_model)
        self.seg_dropout = nn.Dropout(dropout)
        self.pos_emb = PositionalEncoding(d_model, seq_len, dropout)

    def forward(self, tokens: torch.Tensor, seg: torch.Tensor):
        """
        tokens: [B, S]
        seg: [B, S]. seg is binary tensor. 0 indicates that the corresponding token for its index belongs sentence A
        """
        tok_emb = self.tok_emb(tokens) # [B, S, d_model]
        seg_emb = self.seg_emb(seg) # [B, S, d_model]
        pos_emb = self.pos_emb(tokens) # [B, S, d_model]

        return self.tok_dropout(tok_emb) + self.seg_dropout(seg_emb) + pos_emb  # [B, S, d_model]

class Encoder(nn.Module):
    def __init__(self, d_model: int, d_ff:int, seq_len: int, num_enc_heads: int, dropout: float):
        super(Encoder, self).__init__()

        self.attn = MultiHeadAttn(d_model, num_enc_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.fc = PositionWiseFC(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, pad_mask):
        # x+SubLayer(LayerNorm(x))
        layer_norm_x = self.layer_norm1(x)
        attn_score = self.attn(layer_norm_x, layer_norm_x, layer_norm_x, pad_mask) + x
        attn_score = self.dropout1(attn_score)

        # x+SubLayer(LayerNorm(x))
        attn_score = self.fc(self.layer_norm2(attn_score)) + attn_score
        attn_score = self.dropout2(attn_score)

        return attn_score # [B, S, D_model]

class MultiHeadAttn(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_type: str='dot', dropout: float=0.1):
        super(MultiHeadAttn, self).__init__()

        self.attn_type = attn_type.lower()
        assert attn_type in ['dot', 'additive'], 'attn_type not supported: attn_type. Please select one of [dot, addictive]'
        
        assert d_model % num_heads == 0, 'the num_heads doesn\'t match'

        self.num_heads = num_heads
        self.d_k = int(d_model / num_heads)

        # Implements weights via `nn.Linear` instead `nn.Parameters()`
        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

        if self.attn_type == 'additive':
            self.fc = nn.Linear(2*self.d_k, self.d_k)
            self.v = nn.Parameter(torch.rand(1, self.d_k))

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, k, v, q, mask):
        # k, v, q: [B, S, D_model]
        # mask: [B, S, S]
        attn_score = self.scaled_dot_product_attn(q, k, v, mask) # [B, S, D_model]
        attn_score = self.W_o(attn_score) # # [B, S, D_model]

        return attn_score

    def scaled_dot_product_attn(self, k, v, q, mask):
        # k, q, v: [B, S, D_model]
        # mask: [B, S, D_model]
        # FIXME: Attention에서도 사용할 수 있게 re-use 가능한가? -> 
        batch_size, seq_len, _ = k.size()
        q = self.W_q(q).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # [B, h, S, d_k]
        k = self.W_k(k).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # [B, h, S, d_k]
        v = self.W_v(v).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # [B, h, S, d_k]

        if self.attn_type == 'dot':
            qk = self.dot_attn(q, k)
        else:
            qk = self.additive_attn(q, k)
        
        # masking
        if mask != None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1) # [B, h, S, S]
            qk.masked_fill_(mask == True, -1e9)

        attn_score = self.softmax(qk) # [B, h, S, S]
        attn_score = attn_score.matmul(v) # [B, h, S, d_k] 
        attn_score = attn_score.transpose(1, 2).reshape(batch_size, seq_len, self.d_k * self.num_heads) # [B, S, d_model]
        attn_score = self.dropout(attn_score)

        return attn_score
    
    def dot_attn(self, q, k):
        # $Attention(Q, K, V) = softmax(QK^T/\sqrt(d_k))V
        # q, k: # [B, h, S, d_k]
        qk = q.matmul(k.transpose(2, 3)) / torch.sqrt(torch.tensor(self.d_k, dtype=float)) # [B, h, S, S]
        return qk

    def additive_attn(self, q, k):
        # q, k: # [B, h, S, d_k]
        # Neural Machine Translation by Jointly Learning to Align and Translate
        # a(q, k) = w^T * tanh(W*[q:k])
        qk = torch.cat([q, k], dim=-1) # [B, h, S, 2*d_k]
        qk = self.fc(qk) # [B, h, S, d_k]
        qk = qk.permute(0, 1, 3, 2) # [B, d_k, S, d_k]
        v = self.v.reshape(1, 1, 1, -1) # [1, self.d_k] -> [1, 1, 1 self.d_k]
        v = v.repeat(q.size(0), -1, self.num_heads, 1).transpose(1, 2) # [B, h, S, self.d_k]
        qk = v.matul(qk) # [B, h, S, S]

        return qk

class PositionWiseFC(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1):
        super(PositionWiseFC, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # [B, S, D]
        x = self.fc1(x) # [B, S, D_ff]
        x = self.gelu(x) # [B, S, D_ff]
        x = self.fc2(x) # [B, S, D]
        x = self.dropout(x)

        return x

# reference: https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder)_Torch.ipynb
def get_attn_pad_mask(seq_q, seq_k, pad_idx):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1)  # [B, 1, S]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [B, S_q, S_k]