import torch
import torch.nn as nn
import torch.optim as optim
from character_dataset import BPTTIterator, gen_bptt_iter, gen_language_model_corpus
from elmo import ELMo
from trainer import train, evaluate, epoch_time, count_parameters
import math
import matplotlib.pyplot as plt

# MODEL HYPER-PARAMETERS
PRJ_DIM = 512
CNN_DIM = 2048
HID_DIM = 512 # since 4096 / 2 (i.e. # layers) * 2 (i.e. hidden and cell)
CHAR_EMB_DIM = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() == True else 'cpu')
BATCH_SIZE = 32
CHAR_LEN = 50
N_LAYERS = 2
SEQ_LEN = 30

FILTERS = [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]

# get dataset
from torchtext.datasets import WikiText2
from torchtext.data import Field

PAD_WORD = '<pad>'
SOS_WORD = '<sow>'
EOS_WORD = '<eow>'

# The list contains train, valid and test dataset. Each tuple in the list is the dataset of word tokens and the dataset of character tokens
datasets, field_word, field_char = gen_language_model_corpus(WikiText2)
train_data, valid_data, test_data = datasets

VOCAB_DIM = len(field_char.vocab)
OUTPUT_DIM = len(field_word.vocab)

# OTHER HYPER-PARAMETERS
BATCH_SIZE = 32
N_EPOCHS = 100
CLIP = 1
best_valid_loss = float('inf')
# PAD_IDX = field_word.vocab.stoi["<pad>"] # PAD token for word, NOT CHAR

model = ELMo(VOCAB_DIM, OUTPUT_DIM, CHAR_EMB_DIM, HID_DIM, PRJ_DIM, FILTERS, CHAR_LEN, N_LAYERS).to(DEVICE)

# Initialize
model.init_weights()

print(f'The model has {count_parameters(model):,} trainable parameters')

import time

# criterion = cal_loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
train_losses = []
test_losses = []

for epoch in range(1, N_EPOCHS+1):
    train_iter = gen_bptt_iter(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)
    valid_iter = gen_bptt_iter(valid_data, BATCH_SIZE, SEQ_LEN, DEVICE)

    start_time = time.time()
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    train_losses.append(train_loss)
    test_losses.append(valid_loss)

test_iter = gen_bptt_iter(test_data, BATCH_SIZE, SEQ_LEN, DEVICE)
test_loss = evaluate(model, test_iter, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# plot
plt.plot(train_losses, 'b')
plt.plot(test_losses)

plt.show()