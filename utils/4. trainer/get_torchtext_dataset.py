from cgi import test
from torchtext.datasets import IWSLT, WikiText103
from torchtext.data import Field
from typing import Tuple

def get_IWSLT(SEQ_LEN: int=40, exts: Tuple[str, str]=('de', 'en')):
    SRC = Field(tokenize='spacy', tokenizer_language=exts[0], init_token='<SOS>', eos_token='<EOS>', lower=True, batch_first=True, fix_length=SEQ_LEN)
    TRG = Field(tokenize="spacy", tokenizer_language=exts[1], init_token='<SOS>', eos_token='<EOS>', lower=True, batch_first=True, fix_length=SEQ_LEN)

    # change data if want to use other dataset
    train_data, valid_data, test_data = IWSLT.splits(exts=['.'+ext for ext in exts], fields=(SRC, TRG))

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    return (SRC, TRG), (train_data, valid_data, test_data)

def get_WikiText103(SEQ_LEN: int=None, tokenizer_language: str='en'):
    field = Field(tokenize='spacy', tokenizer_language=tokenizer_language, 
                init_token='<SOS>', eos_token='<EOS>', 
                lower=True, batch_first=True, fix_length=SEQ_LEN)

    train_data, valid_data, test_data = IWSLT.splits(text_field=field)

    field.build_vocab(train_data, min_freq=2)

    return field, (train_data, valid_data, test_data)