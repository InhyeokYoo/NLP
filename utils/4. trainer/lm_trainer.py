from typing import Optional, Tuple, Any
import time
from . import utils
from . import get_torchtext_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator

def train(model: nn.Module, iterator: BucketIterator, optimizer: optim.Optimizer, 
        criterion: nn.Module, clip: Optional[float], **kwargs):
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
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def init_weights(model: nn.Module, initializer=torch.nn.init.xavier_uniform_, **kwargs):
    '''
    initialize the weights (default is Xavier initialize). 
    Keyword arguments are for a intializer
    '''
    for p in model.parameters():
        if p.dim() > 1:
            initializer(p, **kwargs)

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

def train(model: nn.Module, criterion: nn.Module=nn.CrossEntropyLoss, optimizer=optim.Adam,
        seq_len: Optional[int]=None, batch_size: int=128, n_epochs: int=100, 
        clip: Optional[float]=None, ignore_idx: Optional[int]=None,
        device = torch.device('cuda' if torch.cuda.is_available() == True else 'cpu')):
    # TODO: MAX_LEN 가져오기
    '''
    trainer for NMT
    param:
        seq_len: Sequence length for the model. the default value is max length of the src/trg corpus
        clip: Optional. If provided, clipping the gradient of the model. (default=None)
        ignore_idx: ignore when calculate the loss. The default value is padding index.
    '''
    SRC, TRG, train_data, valid_data, test_data = get_torchtext_dataset.get_IWSLT(seq_len)

    ENC_PAD_IDX = SRC.vocab.stoi['<pad>']
    DEC_PAD_IDX = TRG.vocab.stoi['<pad>']

    model.to(device)
    # initialize: Xavier
    init_weights(model)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), 
                                                                        batch_size=batch_size, device=device)
    
    criterion(ignore_index=DEC_PAD_IDX)
    optimizer = optimizer(model.parameters())

    train_losses = []
    test_losses = []

    for epoch in range(1, n_epochs+1):
        # lr = D_MODEL**(-1/2) * min(epoch **(-0.5), epoch * WARMUP_STEMPS ** (-1.5))
        # optimizer = optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.98], eps=1e-09)
        
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        utils.show_train_info(epoch, start_time, end_time, train_loss, valid_loss)
        train_losses.append(train_loss)
        test_losses.append(valid_loss)

    test_loss = evaluate(model, test_iterator, criterion)
    utils.show_evaluate_loss(test_loss)
    utils.plot_losses(train_losses, test_losses) # red for train, blue for test

if '__name__' == '__main__':
    # create model
    model = model()
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # fill the parameters!
    train(model)