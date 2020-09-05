# Utils

This repository contains the useful implementations for NLP deep learning archtecture, NLP pre/post processing, tutorials for torchtext, etc. The codes in the repository is implemented via PyTorch.

## List of the implementation

1. torchtext
Jupyternotebook explaining torchtext and the tutorials for:
- `Field`
- `Vocab`
- `Iterator`
- `DataSet`

2. dataset
Modules for creating dataset.

3. dataloader
Modules of dataloaders for text data set. The modules are especially useful when you using TPU of GCP.

4. trainer
Modules for training NLP archtecture via torchtext. The modules are including:
- Neural Machine Trasnlation
    - [IWSLT 2016](https://sites.google.com/site/iwsltevaluation2016/)
- Language Modeling
    - [WikiText103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/#published-results-wikitext-103)