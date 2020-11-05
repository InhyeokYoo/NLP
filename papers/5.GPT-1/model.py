import torch
import torch.nn as nn

class LanguageModeling(nn.Module):
    # Unsupervised pre-training
    def __init__(self, voc_size:int, seq_len: int, d_model: int, d_ff:int, pad_idx: int,
                num_decoder: int, num_heads: int, dropout: float) -> None:
        super(LanguageModeling, self).__init__()
        self.pad_idx = pad_idx

        self.decoders = Decoders(voc_size, seq_len, d_model, d_ff, pad_idx, num_decoder, num_heads, dropout)

        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        param:
            input: sequence of words
        dim:
            input:
                input: [B, S]
            output:
                result: [B, S, D_model]
        '''
        pad_mask = get_attn_pad_mask(input, input, self.pad_idx)
        attn = self.decoders(input, pad_mask) # [B, S, D_model]
        output = self.linear(attn) # [B, S, voc_size]

        return output # [B, S, voc_size]

class Decoders(nn.Module):
    def __init__(self, voc_size:int, seq_len: int, d_model: int, d_ff: int, pad_idx: int,  
                num_decoder: int, num_heads: int, emb_dropout: float, dec_dropout: float) -> None:
        super(Decoders, self).__init__()
        self.d_model = d_model
        self.voc_size = voc_size
        self.pad_idx = pad_idx

        self.emb = nn.Embedding(num_embeddings=voc_size, embedding_dim=d_model)
        self.dropout = nn.Dropout(emb_dropout)
        # positional embedding is trainable
        self.pos_emb = nn.Embedding(num_embeddings=seq_len+1, embedding_dim=d_model)
        self.pos_dropout = nn.Dropout(emb_dropout)
        self.models = nn.ModuleList([Decoder(d_model, d_ff, seq_len, num_heads, dropout=dec_dropout) for i in range(num_decoder)])

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, input: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        '''
        param:
            input: sequence of words
        dim:
            input:
                input: [B, S]
            output:
                result: [B, S, D_model]
        '''
        batch_size, seq_len = input.size()
        # embedding
        emb = self.dropout(self.emb(input)) # [B, S, D_model]
        position = torch.arange(seq_len, device=input.device, dtype=input.dtype).repeat(batch_size, 1) # [B, S, D_model]
        emb += self.pos_dropout(self.pos_emb(position)) # [B, S, D_model]

        # masking
        subsequent_pad_mask = get_attn_subsequent_mask(emb)
        mask = pad_mask + subsequent_pad_mask

        for model in self.models:
            input = model(input, mask)

        return input

class Decoder(nn.Module):
    def __init__(self, d_model: int, d_ff:int, num_heads: int, dropout: float) -> None:
        super(Decoder, self).__init__()

        self.self_attn = MultiHeadAttn(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.fc = PositionWiseFC(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        param:
            input: input tensor. Embedded words
            mask: Subsequence mask of decoders. masking if > 1
        dim:
            input: [Batch, Seq_len, D_embedding]
            mask: [Batch, Seq len(q), Seq_len(k)]
            attn_score: [Batch, Seq len(q), Seq_len(k)]
        '''
        # decoder self attention: x+SubLayer(LayerNorm(x))
        layer_norm_input = self.layer_norm1(input)
        self_attn_score = input + self.self_attn(layer_norm_input, layer_norm_input, layer_norm_input, mask)
        self_attn_score = self.dropout1(self_attn_score)

        # Position-wise feed forward network: x+SubLayer(LayerNorm(x))
        attn_score = self.fc(self.layer_norm2(self_attn_score)) + self_attn_score
        attn_score = self.dropout2(attn_score)
        
        return attn_score

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
    def __init__(self, d_model: int, d_ff: int, active: nn.Module=nn.GELU, dropout: float=0.1):
        super(PositionWiseFC, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.active = active()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        # [B, S, D]
        output = self.active(self.fc1(input)) # [B, S, D_ff]
        output = self.dropout(self.fc2(output)) # [B, S, D]

        return output

# Supervised fine-tuning
class ClassificationHead(nn.Module):
    def __init__(self, model: nn.Module, ) -> None:
        self.model = model
        self.linear = nn.Linear()

    def forward(self, input) -> torch.Tensor:
        pass

class EntailmentHead(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self) -> torch.Tensor:
        pass

class SimilarityHead(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self) -> torch.Tensor:
        pass

class MultipleChoiceHead(nn.Module):
    def __init__(self) -> None:
        pass

    def forward(self) -> torch.Tensor:
        pass

# reference: https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder)_Torch.ipynb
def get_attn_pad_mask(seq_q: torch.Tensor, seq_k: torch.Tensor, pad_idx: int):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1)  # [B, 1, S]
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [B, S_q, S_k]

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] # [B, S, S]
    subsequent_mask = torch.zeros(attn_shape)
    subsequent_mask.triu_(1) # [B, S, S]
    return subsequent_mask

print(True)