# Readme.md

An implementation of [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding review](https://arxiv.org/abs/1810.04805) of Devlin et al. (2018).

**TODO LIST**:
- replace i-th token with the [MASK]token 80% of the time, a random token 10% of the time, ann the unchanged i-th token 10% of the time
- shuffling datasets
- dealing a combination of two sequences, of which has over 512 length
- warm up optimizer
- pre-train  the  model  with  sequence  length  of 128 for 90% of the steps. The rest 10% of the steps of sequence of 512.
- the fine-tuning tasks

# Usage

In `train.py`, fill the `args` for BERT model.
I skip detail explanation since the names of the variables are very intuitive.

- dataset: arguments for datasets
- wordpiece: arguments for `sentencepiece`
- model: 

```python
# Args:
    args = {
        'dataset': {
            "train_input_file": "/content/drive/MyDrive/Colab-Notebooks/datasets/BookCorpus_train.txt",
            "test_input_file": "/content/drive/MyDrive/Colab-Notebooks/datasets/BookCorpus_test.txt",
            "val_input_file": "/content/drive/MyDrive/Colab-Notebooks/datasets/BookCorpus_val.txt"
            },
        'wordpiece': {
            "vocab_size": 30000,
            "prefix": 'bookcorpus_spm',
            'user_defined_symbols': '[PAD],[CLS],[SEP],[MASK]',
            'model_type': 'bpe',
            'character_coverage': 1.0, # default
        },
        'model': {
            "seq_len":512,
            "mask_frac":0.15,
        },
        'train': {
            "batch_size": 24,
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "weight_decay": 0.01,
            "weight_decay": 0.01,
            "device": torch.device('cuda' if torch.cuda.is_available() == True else 'cpu'),
        }
    }
```



# Experimental

## Result