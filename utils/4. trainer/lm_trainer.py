from typing import Optional, Tuple
import math
import time
from . import utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IWSLT
from torchtext.data import Field, BucketIterator

def train(model: nn.Module, iterator: BucketIterator, optimizer: optim.Optimizer, 
        criterion: nn.Module, clip: float, **kwargs):
    model.train()
    epoch_loss = 0

    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)

        output = output[:, 1:].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg, **kwargs)
        loss.backward()

        # Gradient clipping
        if clip != None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def init_weights(model: nn.Module):
    # Xavier initialize
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def evaluate(model: nn.Module, iterator: BucketIterator, criterion: nn.Module,
            **kwargs):
    # kwargs for the criterion (e.g. Label Smoothing)
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg, **kwargs)
            # loss = criterion(output, trg, PAD_IDX, smoothing=0.1) # Label smoothing
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def get_torchtext_dataset(SEQ_LEN: int=40, exts: Tuple[str, str]=('de', 'en')):
    SRC = Field(tokenize='spacy', tokenizer_language=exts[0], init_token='<SOS>', eos_token='<EOS>', lower=True, batch_first=True, fix_length=SEQ_LEN)
    TRG = Field(tokenize="spacy", tokenizer_language=exts[1], init_token='<SOS>', eos_token='<EOS>', lower=True, batch_first=True, fix_length=SEQ_LEN)

    # change data if want to use other dataset
    train_data, valid_data, test_data = IWSLT.splits(exts=['.'+ext for ext in exts], fields=(SRC, TRG))

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    return (SRC, TRG), (train_data, valid_data, test_data)

if '__name__' == '__main__':
    # get data_set: IWSLT from torchtext
    SRC, TRG, train_data, valid_data, test_data = get_torchtext_dataset(SEQ_LEN)

    # FINALS
    SEQ_LEN = 40
    ENC_PAD_IDX = SRC.vocab.stoi['<pad>']
    DEC_PAD_IDX = TRG.vocab.stoi['<pad>']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() == True else 'cpu')
    BATCH_SIZE = 128
    # Model **kwargs
    kwargs = {'src_size': len(SRC.vocab), 
              'trg_size': len(TRG.vocab),
              'BATCH_SIZE': BATCH_SIZE,
                'D_MODEL': 512,
                'D_FF': 2048,
                'DEVICE': DEVICE,
                'ENC_PAD_IDX':ENC_PAD_IDX,
                'DEC_PAD_IDX':DEC_PAD_IDX,
              }

    # create model
    model = model(**kwargs).to(DEVICE)

    # initialize: Xavier
    init_weights(model)
    N_EPOCHS = 100
    CLIP = 1
    
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=DEVICE)

    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    criterion = nn.CrossEntropyLoss(ignore_index=DEC_PAD_IDX)
    optimizer = optim.Adam(model.parameters())

    train_losses = []
    test_losses = []

    for epoch in range(1, N_EPOCHS+1):
        # lr = D_MODEL**(-1/2) * min(epoch **(-0.5), epoch * WARMUP_STEMPS ** (-1.5))
        # optimizer = optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.98], eps=1e-09)
        
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        utils.show_train_info(epoch, start_time=start_time, end_time=end_time,
                            train_loss=train_loss, valid_loss=valid_loss)
        train_losses.append(train_loss)
        test_losses.append(valid_loss)

    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    utils.plot_losses(train_losses, test_losses) # red for train, blue for test
`