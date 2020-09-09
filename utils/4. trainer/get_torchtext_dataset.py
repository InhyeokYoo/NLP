from torchtext.data import Field
from typing import Tuple

def get_IWSLT(seq_len: int=40, exts: Tuple[str, str]=('de', 'en')):
    from torchtext.datasets import IWSLT
    SRC = Field(tokenize='spacy', tokenizer_language=exts[0], init_token='<SOS>', eos_token='<EOS>', lower=True, batch_first=True, fix_length=seq_len)
    TRG = Field(tokenize="spacy", tokenizer_language=exts[1], init_token='<SOS>', eos_token='<EOS>', lower=True, batch_first=True, fix_length=seq_len)

    # change data if want to use other dataset
    train_data, valid_data, test_data = IWSLT.splits(exts=['.'+ext for ext in exts], fields=(SRC, TRG))

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    return (SRC, TRG), (train_data, valid_data, test_data)

def get_WikiText103(seq_len: int=None, tokenizer_language: str='en'):
    from torchtext.datasets import WikiText103
    field = Field(tokenize='spacy', tokenizer_language=tokenizer_language, 
                init_token='<SOS>', eos_token='<EOS>', 
                lower=True, batch_first=True, fix_length=seq_len)

    train_data, valid_data, test_data = WikiText103.splits(text_field=field)

    field.build_vocab(train_data, min_freq=2)

    return field, (train_data, valid_data, test_data)