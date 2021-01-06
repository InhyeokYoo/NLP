import random
import torch
from typing import List
import random
import sentencepiece as spm

class BERTLanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, data: List, vocab: spm.SentencePieceProcessor, sep_id: str='[SEP]', cls_id: str='[CLS]',
                mask_id: str='[MASK]', pad_id: str="[PAD]", seq_len: int=512, mask_frac: float=0.15, p: float=0.5):
        """
        Initiate language modeling dataset for BERT.
        
        params:
            data (list): a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab (sentencepiece.SentencePieceProcessor): Vocabulary object used for dataset.
            p (float): probability for NSP. defaut 0.5

        example:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=48, shuffle=False)
        """
        super(BERTLanguageModelingDataset, self).__init__()
        self.vocab = vocab
        self.data = data
        self.seq_len = seq_len
        self.sep_id = vocab.piece_to_id(sep_id)
        self.cls_id = vocab.piece_to_id(cls_id)
        self.mask_id = vocab.piece_to_id(mask_id)
        self.pad_id = vocab.piece_to_id(pad_id)
        self.p = p
        self.mask_frac = mask_frac

    def __getitem__(self, i):
        # TODO
        # add padding
        # sentence A, B
        # sentence embedding for pad: 어차피 마스킹되서 상관없나..?
        # 512보다 작은거 해결
        # 90% 128, 10% 512

        seq1 = self.vocab.EncodeAsIds(self.data[i].strip())
        seq2_idx = i+1
        # decide wheter use random next sentence or not
        if random.random() > self.p:
            is_next = torch.tensor(1)
            while seq2_idx == i+1:
                seq2_idx = random.randint(0, len(self.data))
        else:
            is_next = torch.tensor(0)

        seq2 = self.vocab.EncodeAsIds(self.data[seq2_idx])

        if len(seq1) + len(seq2) >= self.seq_len - 3: # except 1 [CLS] and 2 [SEP]
            # 만약 seq1만으로 512 달성하면 어떻게 되지?
            # 만약 seq1 + seq2가 512보다 크면 어떡하지?
            idx = self.seq_len - 3 - len(seq1)
            seq2 = seq2[:idx]

        # sentence embedding: 0 for A, 1 for B
        mlm_target = torch.tensor([self.cls_id] + seq1 + [self.sep_id] + seq2 + [self.sep_id] + [self.pad_id] * (self.seq_len - 3 - len(seq1) - len(seq2))).long().contiguous()
        sent_emb = torch.ones((mlm_target.size(0)))
        _idx = len(seq1) + 2
        sent_emb[:_idx] = 0
        
        def masking(data):
            data = torch.tensor(data).long().contiguous()
            data_len = data.size(0)
            ones_num = int(data_len * self.mask_frac)
            zeros_num = data_len - ones_num
            lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
            lm_mask = lm_mask[torch.randperm(data_len)]
            data = data.masked_fill(lm_mask.bool(), self.mask_id)

            return data

        mlm_train = torch.cat([torch.tensor([self.cls_id]), masking(seq1), torch.tensor([self.sep_id]), masking(seq1), torch.tensor([self.sep_id])]).long().contiguous()
        mlm_train = torch.cat([mlm_train, torch.tensor([self.pad_id] * (512 - mlm_train.size(0)))]).long().contiguous()

        # mlm_train, mlm_target, sentence embedding, NSP target
        return mlm_train, mlm_target, sent_emb, is_next

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab

    def decode(self, x):
        return self.vocab.DecodeIds(x)

class QADataset(torch.utils.data.Dataset):
    def __init__(self, data: List, vocab: spm.SentencePieceProcessor, sep_id: str='[SEP]', cls_id: str='[CLS]',
                mask_id: str='[MASK]', pad_id: str="[PAD]", seq_len: int=512, mask_frac: float=0.15, p: float=0.5):
        """
        Initiate language modeling dataset for BERT.
        
        params:
            data (list): a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab (sentencepiece.SentencePieceProcessor): Vocabulary object used for dataset.
            p (float): probability for NSP. defaut 0.5

        example:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=48, shuffle=False)
        """
        super(BERTLanguageModelingDataset, self).__init__()
        self.vocab = vocab
        self.data = data
        self.seq_len = seq_len
        self.sep_id = vocab.piece_to_id(sep_id)
        self.cls_id = vocab.piece_to_id(cls_id)
        self.mask_id = vocab.piece_to_id(mask_id)
        self.pad_id = vocab.piece_to_id(pad_id)
        self.p = p
        self.mask_frac = mask_frac

    def __getitem__(self, i):
        # TODO
        # add padding
        # sentence A, B
        # sentence embedding for pad: 어차피 마스킹되서 상관없나..?
        # 512보다 작은거 해결
        # 90% 128, 10% 512

        seq1 = self.vocab.EncodeAsIds(self.data[i].strip())
        seq2_idx = i+1
        # decide wheter use random next sentence or not
        if random.random() > self.p:
            is_next = torch.tensor(1)
            while seq2_idx == i+1:
                seq2_idx = random.randint(0, len(self.data))
        else:
            is_next = torch.tensor(0)

        seq2 = self.vocab.EncodeAsIds(self.data[seq2_idx])

        if len(seq1) + len(seq2) >= self.seq_len - 3: # except 1 [CLS] and 2 [SEP]
            # 만약 seq1만으로 512 달성하면 어떻게 되지?
            # 만약 seq1 + seq2가 512보다 크면 어떡하지?
            idx = self.seq_len - 3 - len(seq1)
            seq2 = seq2[:idx]

        # sentence embedding: 0 for A, 1 for B
        mlm_target = torch.tensor([self.cls_id] + seq1 + [self.sep_id] + seq2 + [self.sep_id] + [self.pad_id] * (self.seq_len - 3 - len(seq1) - len(seq2))).long().contiguous()
        sent_emb = torch.ones((mlm_target.size(0)))
        _idx = len(seq1) + 2
        sent_emb[:_idx] = 0
        
        def masking(data):
            data = torch.tensor(data).long().contiguous()
            data_len = data.size(0)
            ones_num = int(data_len * self.mask_frac)
            zeros_num = data_len - ones_num
            lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
            lm_mask = lm_mask[torch.randperm(data_len)]
            data = data.masked_fill(lm_mask.bool(), self.mask_id)

            return data

        mlm_train = torch.cat([torch.tensor([self.cls_id]), masking(seq1), torch.tensor([self.sep_id]), masking(seq1), torch.tensor([self.sep_id])]).long().contiguous()
        mlm_train = torch.cat([mlm_train, torch.tensor([self.pad_id] * (512 - mlm_train.size(0)))]).long().contiguous()

        # mlm_train, mlm_target, sentence embedding, NSP target
        return mlm_train, mlm_target, sent_emb, is_next

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab

    def decode(self, x):
        return self.vocab.DecodeIds(x)
