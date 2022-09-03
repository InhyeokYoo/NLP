# Readme.md

An implementation of the model [Attention is all you need](https://arxiv.org/abs/1706.03762) of Vaswani et al. (2017).

I implement transformer archtecture, label smoothing, beam search and warm-up steps. There are several things that differ to the original paper.

# Experiment

I compare the effects of label smoothing, layer normalization, warm-up steps, etc. of the original paper.

## Changing the location of SubLayer and LayerNorm

The figure below is the train (blue) and validation (sky blue) losses of the model using **x+SubLayer(LayerNorm(x))** instead of the original paper's LayerNorm(x + Sublayer(x)).

![image](https://user-images.githubusercontent.com/47516855/91271179-bed54980-e7b4-11ea-9419-3e24c0e697a2.png)

The error start with very low point for both train/valid set.
```
# Errors at 1-st epoch
Train Loss: 0.676 | Train PPL:   1.966
Val. Loss: 0.121 |  Val. PPL:   1.128
```

We can see that the model achieves best performance early.

One thing worth to note is that **the validation losses are always lower** than the training loss during entire training process. I'm not sure why does this happen.

## Adapting Warm-up steps and Label smoothing

The figure below is the train (blue) and validation (sky blue) losses of the model that adapts **warm-up steps and label smoothing** of the original paper.
We can see that the losses are high at very first but they are getting better thorugh the iteration. Unlike previous model, the training losses are always lower than validation losses.

![index](https://user-images.githubusercontent.com/47516855/91434020-2cf63b00-e89f-11ea-9898-32c798029ce6.png)

---

The table shows the result of the models of this repo. I use [IWSLT](https://pytorch.org/text/datasets.html#torchtext.datasets.IWSLT.splits) dataset for NMT in torchtext

The epoch is set to 100 and the seed is not fixed in the experiment. Therefore, the models are not able to be compared the performances of them but you can reference the result.

| Model | Train Loss | Valid Loss | Test Loss |
| --- | :---: | :---: | :---: |
| Baseline | 5.174 | 5.773 | 5.936 |
| x+SubLayer(LayerNorm(x)) | **0.031** | **0.002** | **0.013** |
| Label smoothing + Warm-up steps | 1.555 | 1.687 | 1.739 |
