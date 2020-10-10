import torch
import torch.nn as nn
import torch.optim as optim
from character_dataset import BPTTIterator


def count_parameters(model: nn.Module):
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