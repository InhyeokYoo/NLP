import torch.nn as nn
import torch
import torch.optim as optim
from torchtext.datasets import BPTTIterator
from model import Decoders

class LanguageModeling(nn.Module):
    def __init__(self, voc_size:int, seq_len: int, d_model: int, d_ff:int, 
                num_decoder: int, num_heads: int, dropout: float) -> None:
        super(LanguageModeling, self).__init__
        self.decoders = Decoders(voc_size, seq_len, d_model, d_ff, num_decoder, num_heads, dropout)
        self.softmax = nn.softmax(dim=-1)
        
    def forward(self, input: torch.Tensor):
        '''
        param:
            input: sequence of words
        dim:
            input:
                input: [B, S]
            output:
                result: [B, S, D_model]
        '''
        emb = self.decoders(input) # [B, S, D_model]
        result = self.softmax(emb) # [B, S, D_model]

        return result

def count_parameters(model: nn.Module):
    # count # params of the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model: nn.Module, iterator: BPTTIterator, optimizer: optim.Optimizer, criterion: nn.Module, clip: float):
    model.train()
    epoch_loss = 0

    cnt = 0 # count length for avg loss
    for _, (_, char_text, word_target, _) in enumerate(iterator): # (word_text, char_text, word_target, char_taget)
        src = char_text
        trg = word_target

        optimizer.zero_grad()

        output = model(src)
        output = output.reshape(-1, output.shape[-1])
        trg = trg.reshape(-1)
        loss = criterion(output, trg) # CE
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        cnt += 1

    return epoch_loss / cnt

def evaluate(model: nn.Module, iterator: BPTTIterator, criterion: nn.Module):
    model.eval()
    epoch_loss = 0

    cnt = 0 # count length for avg loss
    with torch.no_grad():
        for _, (_, char_text, word_target, _) in enumerate(iterator): # word_text, char_text, word_target, char_taget
            src = char_text
            trg = word_target
            
            output = model(src)

            output = output.reshape(-1, output.shape[-1])
            trg = trg.reshape(-1)
            loss = criterion(output, trg) # CE
            epoch_loss += loss.item()
            cnt += 1

    return epoch_loss / cnt

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs