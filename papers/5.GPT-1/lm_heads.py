import torch, torch.nn as nn

class LMHead(nn.Module):
    def __init__(self, lm: nn.Module) -> None:
        voc_size, d_model = lm.decoders.emb.weight.size() # voc_size, d_model
        self.lm = lm
        # TODO: 논문에 의하면 embedding matrix 다시 사용: weight만 share?
        # TODO: 왜 bias가 없는지 모르겠음
        self.linear = nn.Linear(d_model, voc_size, bias=False)
        self.linear.weight = lm.decoders.emb.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attn = self.lm(input)
        output = self.linear(attn)

        return output

class ClfHead(nn.Module):
    def __init__(self, lm: nn.Module, n_classes: int, dropout: float=0.1) -> None:
        self.lm = lm
        self.linear = (lm.emb.weight.size(0), n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.lm(input)
        output = self.linear(output)
        
        return self.dropout(output)

class QAHead(nn.Module):
    def __init__(self, lm: nn.Module, ) -> None:
        self.lm = lm