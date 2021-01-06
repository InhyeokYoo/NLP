# NLP

This repository contains implementations of NLP deep learning archtecture. The implementation in this repository is implemented via PyTorch. The list of the models is on the below:

# List of the implementation

## 1. Attention

Implementations of Bahdanau, D., Cho, K., & Bengio, Y. (2014) (https://arxiv.org/abs/1409.0473) and Luong, M. T., Pham, H., & Manning, C. D. (2015) (https://arxiv.org/abs/1508.04025)

- Bahdanau Attention
- Luong Attention
    - global attention
    - local attention
    
## 2. Sub-word Model

No implementation. Just simple tutorials for sub-word models.

- Byte Pair Encoding (BPE)
- SentencePiece

## 3. Self-attention (?? - 2020 Sep 04) 

An implementation of the model [Attention is all you need](https://arxiv.org/abs/1706.03762) of Vaswani., et al. (2017), aka Transformer.
I implement transformer archtecture, label smoothing, beam search and warm-up steps. There are several things that differ to the original paper.

## 4. ELMo (2020 Sep 04 - 2020 Oct 09)

An implementation of [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) of Peters et al. (2018). 
The module also includes character embedding of [Kim et al. (2015)](https://arxiv.org/pdf/1508.06615.pdf).

- ELMo
- Character-Aware Neural Language Models

## 5. GPT-1 (2020 Oct 16 - )

## 6. BERT (2020 Nov 12 - 2021 Jan 03)

