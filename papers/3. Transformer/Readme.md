# Readme.md

The figure below is the train (blue) and validation (sky blue) losses of the model using **x+SubLayer(LayerNorm(x))** instead of the original paper's LayerNorm(x + Sublayer(x)). However, the model shows lower error on validation set than training set.

![image](https://user-images.githubusercontent.com/47516855/91271179-bed54980-e7b4-11ea-9419-3e24c0e697a2.png)

---

The figure below is the train (blue) and validation (sky blue) losses of the model that adapts **warm-up steps and label smoothing** of the original paper.
The losses are high at very first but they are getting better thorugh the iteration.

![index](https://user-images.githubusercontent.com/47516855/91434020-2cf63b00-e89f-11ea-9898-32c798029ce6.png)

| Test Loss: 1.739 | Test PPL:   5.689 |


# Usage