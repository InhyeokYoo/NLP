# Reference: https://gist.github.com/akurniawan/30719686669dced49e7ced720329a616
import itertools
import torch
from torchtext.experimental.datasets.translation import DATASETS, TranslationDataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.experimental.functional import sequential_transforms
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def build_char_vocab(
    data,
    transforms,
    index,
    init_word_token="<sow>",
    eos_word_token="<eow>",
    init_sent_token="<s>",
    eos_sent_token="</s>",
):
    tok_list = [
        [init_word_token],
        [eos_word_token],
        [init_sent_token],
        [eos_sent_token],
    ]
    for line in data:
        tokens = list(itertools.chain.from_iterable(transforms(line[index])))
        tok_list.append(tokens)
    return build_vocab_from_iterator(tok_list)

def char_vocab_func(vocab):
    def func(tok_iter):
        return [[vocab[char] for char in word] for word in tok_iter]

    return func

def special_tokens_func(init_word_token="<w>",
                        eos_word_token="</w>",
                        init_sent_token="<s>",
                        eos_sent_token="</s>",):
    
    def func(tok_iter):
        result = [[init_word_token, init_sent_token, eos_word_token]]
        result += [[init_word_token] + word + [eos_word_token] for word in tok_iter]
        result += [[init_word_token, eos_sent_token, eos_word_token]]
        return result

    return func

def pad_chars(input, pad_idx=1):
    # get info on length on each sentences
    batch_sizes = [len(sent) for sent in input]
    # flattening the array first and convert them to tensor
    tx = list(map(torch.tensor, itertools.chain.from_iterable(input)))
    # pad all the chars
    ptx = pad_sequence(tx, True, pad_idx)
    # split according to the original length
    sptx = ptx.split(batch_sizes)
    # finally, merge them back with padding
    final_padding = pad_sequence(sptx, True, pad_idx)

    return final_padding

if __name__ == "__main__":
    # Get the raw dataset first. This will give us the text
    # version of the dataset
    train, test, val = DATASETS["Multi30k"]()
    # Cache training data for vocabulary construction
    train_data = [line for line in train]
    # Setup word tokenizer
    src_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    tgt_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    # Setup char tokenizer

    def char_tokenizer(words):
        return [list(word) for word in words]

    src_char_transform = sequential_transforms(src_tokenizer, char_tokenizer)
    tgt_char_transform = sequential_transforms(tgt_tokenizer, char_tokenizer)

    # Setup vocabularies (both words and chars)
    src_char_vocab = build_char_vocab(train_data, src_char_transform, index=0)
    tgt_char_vocab = build_char_vocab(train_data, tgt_char_transform, index=1)

    # Building the dataset with character level tokenization
    src_char_transform = sequential_transforms(
        src_char_transform, special_tokens_func(), char_vocab_func(src_char_vocab)
    )
    tgt_char_transform = sequential_transforms(
        tgt_char_transform, special_tokens_func(), char_vocab_func(tgt_char_vocab)
    )
    train_dataset = TranslationDataset(
        train_data,
        (src_char_vocab, tgt_char_vocab),
        (src_char_transform, tgt_char_transform),
    )

    # Prepare DataLoader
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        padded_src_batch = pad_chars(src_batch)
        padded_tgt_batch = pad_chars(tgt_batch)
        return (padded_src_batch, padded_tgt_batch)

    train_iterator = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
    for batch in train_iterator:
        src = batch[0]
        tgt = batch[1]
        print(src.size())
        print(tgt.size())