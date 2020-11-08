import torch
import torch.nn as nn

class LMHead(nn.Module):
    '''
    A head for language modeling in the original paper: softmax(h_n, W_e^T).
    The LMHead is trained on both pre-trained step and fine-tuning step.
    '''
    def __init__(self, lm: nn.Module) -> None:
        voc_size, d_model = lm.decoders.emb.weight.size() # voc_size, d_model
        self.lm = lm
        self.linear = nn.Linear(d_model, voc_size, bias=False)
        self.linear.weight = lm.decoders.emb.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attn = self.lm(input)
        output = self.linear(attn)

        return output

class ClfHead(nn.Module):
    '''
    A head for text classificataion task: Stanford Sentiment Treebank-2, CoLA.
    The model expects a dataset of a form of [<s> TEXT <e>].
    '''
    def __init__(self, lm: nn.Module, n_classes: int, dropout: float=0.1) -> None:
        self.lm = lm
        self.linear = nn.Linear(lm.emb.weight.size(1), n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        params:
            input: a sentence to be classified by the model. The sentence includes <sos>, <eos>, <pad> idx.
        dims:
            input: [B, Seq_len]
        '''
        output = self.lm(input)
        output = self.dropout(output)
        output = self.linear(output)
        
        return output

class MultipleChoiceHead(nn.Module):
    '''
    Multiple choice head for question answering (RACE, Story Cloze) 
    and text entailment fine-tuning task (SNLI, MultiNLI, Question NLI, RTE and SciTail).
    In Q.A task, 
        the model expects to get concatenation of [<s>, document, question, <$> possible answers, <e>] as an input.
        And then, the model solves a classification problem for all possible asnwers.
    In text entailment task (i.e. natural language inference), 
        the model expects to get concatenation of [<s>, premise, <$>, hypothesis, <eos>] as an input.
        And then, the model sovles a classification problem that choosing among three different natural language inference 
        classes: entailment, neutral and contradiction.
    '''
    def __init__(self, lm: nn.Module, n_classes: int, dropout: float=0.1) -> None:
        self.lm = lm
        self.linear = nn.Linear(lm.emb.weight.size(1), n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.lm(input)
        output = self.dropout(output)

        return output

class SimilarityHead(nn.Module):
    '''
    A head for comparing sentence similarity between two sentences: 
    MSR Paraphrase Corpus, Quora Question Pairs, STS Benchmark.
    The model expects to get a pair of dataset, [<s>, TEXT1, <$>, TEXT2, <e>] 
    and [<s>, TEXT2, <$>, TEXT1, <e>] as inputs.
    '''
    def __init__(self, lm: nn.Module, n_classes: int, dropout: float=0.1) -> None:
        self.lm = lm
        self.linear = nn.Linear(lm.emb.weight.size(1), n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1: torch.Tensor, input2:torch.Tensor) -> torch.Tensor:
        '''
        param:
            input1: Concatenation of text1 and text2
            input2: Concatenation of text2 and text1
        '''
        output1 = self.lm(input1)
        output2 = self.lm(input2)
        output = output1 + output2
        output = self.dropout(output)
        output = self.linear(output)
        
        return output