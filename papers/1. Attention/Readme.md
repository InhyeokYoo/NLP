# Readme.md

The repo includes implementations of Bahdanau, D., Cho, K., & Bengio, Y. (2014)(https://arxiv.org/abs/1409.0473) and Luong, M. T., Pham, H., & Manning, C. D. (2015) (https://arxiv.org/abs/1508.04025)

## Bahdanau attention

Both encoder and decoder are able to set their RNN as RNN, GRU or LSTM. The dimensions of the hidden states of the encoder and the decoder are not the same. However, the numbers of layers in encoder and decoder are the same. The hidden state (and cell state for LSTM case) of the encoder will be concatenated when `bidirectional=True`, and feed to `torch.tanh(self.fc_h(h))` for matching to the decoder's dimension. The initial hidden state of the encoder will be not provided (i.e. zeros will be provided). The initial hidden state of the decoder is decribed in Appendix 2.2 of the original paper. attn_dim The model supports only 'concat' for the alignment function. The below is an example:

```python
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

EMB_DIM = 32
HID_DIM = 64
ATTN_DIM = HID_DIM // 8
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2

BIDIRECTIONAL = True
NUM_DIR = 2 if BIDIRECTIONAL else 1
TEACHER_FORCING = 0.5
RNN_TYPE = 'LSTM'
DROPOUT = 0.2
N_LAYERS = 1

encoder = Encoder(RNN_TYPE, INPUT_DIM, EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, dropout=DROPOUT, bidirectional=BIDIRECTIONAL, n_layers=N_LAYERS)
attention = Attention(RNN_TYPE, ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM, num_dir=NUM_DIR)
decoder = BahdanauDecoder(RNN_TYPE, OUTPUT_DIM, EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, attention, dropout=DROPOUT, num_dir=NUM_DIR, n_layers=N_LAYERS)
model = Seq2Seq(rnn_type=RNN_TYPE, encoder=encoder, decoder=decoder, trg_vocab_size=OUTPUT_DIM, device=device, num_dir=NUM_DIR, teacher_forcing=TEACHER_FORCING).to(device)
```

## Luong attention

RNN model for bnoth encoder and decoder are able to RNN, GRU and LSTM. The dimensions of the hidden states and the numbers of layers in encoder and decoder are the same. The hidden state (and cell state for LSTM case) of the encoder will be concatenated when `bidirectional=True`, and feed to `torch.tanh(self.fc_h(h))`, and it will be the init state of the decoder. The initial hidden state of the encoder will be not provided (i.e. zeros will be provided). The hidden states of the encoder (<a href="https://www.codecogs.com/eqnedit.php?latex=H" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H" title="H" /></a>, `output`) will be summed along `dim=2`. Both global and local attention are supported. The monotonic and predictive alignment are supported. Every alginment functions that indtroduced in the paper are supported. The below is an example:

```python
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

EMB_DIM = 32
HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2

BIDIRECTIONAL = True
NUM_DIR = 2 if BIDIRECTIONAL else 1
TEACHER_FORCING = 0.5
RNN_TYPE = 'LSTM'
DROPOUT = 0.2
N_LAYERS = 1
D = 10
Pt = 'predictive'
ATTN_TYPE = 'local'
ALIGN_FN = 'concat'

# Luong
encoder = Encoder(RNN_TYPE, INPUT_DIM, EMB_DIM, HID_DIM, dropout=DROPOUT, bidirectional=BIDIRECTIONAL, n_layers=N_LAYERS)
attention = Attention(RNN_TYPE, ALIGN_FN, HID_DIM, num_dir=NUM_DIR)
decoder = LuongDecoder(RNN_TYPE, ATTN_TYPE, OUTPUT_DIM, EMB_DIM, HID_DIM, attention, device=device, dropout=DROPOUT, num_dir=NUM_DIR, n_layers=N_LAYERS, Pt=Pt)
model = Seq2Seq(rnn_type=RNN_TYPE, encoder=encoder, decoder=decoder, trg_vocab_size=OUTPUT_DIM, device=device, num_dir=NUM_DIR, teacher_forcing=TEACHER_FORCING).to(device)
```

