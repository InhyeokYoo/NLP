'''
Reference: https://github.com/kh-kim/simple-nmt/blob/master/simple_nmt/data_loader.py
'''

from torchtext import data, datasets

class DataLoader():

    def __init__(
            self, train, valid, exts, batch_size, device,
            tokenize=None, tokenize_lang=None, max_vocab=99999999, max_length=255, 
            fix_length=None, batch_first=True, shuffle=True):

        super(DataLoader, self).__init__()
        '''
        Params:
            train_fn - path for train data set
            valid_fn - path for valid data set
            exts – A tuple containing the extension to path for each language.
            tokenize - tokenizer. E.g. spacy
            tokenize_lang - A tuple containing srg and trg language

        '''
        assert tokenize != 'SpaCy' and tokenize_lang != None, 'Various languages currently supported only in SpaCy.'

        self.src = data.Field(
                batch_first=batch_first, tokenize=tokenize, tokenizer_language=tokenize_lang[0],
                fix_length=fix_length, init_token='<SOS>', eos_token='<EOS>', lower=True,)
        
        self.tgt = data.Field(
                batch_first=batch_first, tokenize=tokenize, tokenizer_language=tokenize_lang[1],
                fix_length=fix_length, init_token='<SOS>', eos_token='<EOS>', lower=True,)

        self.train_iter = data.BucketIterator(
            train,
            batch_size=batch_size,
            device=device,
            shuffle=shuffle,)

        self.valid_iter = data.BucketIterator(
            valid,
            batch_size=batch_size,
            device=device,
            shuffle=False,)

        self.src.build_vocab(train, max_size=max_vocab)
        self.tgt.build_vocab(train, max_size=max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab

if __name__ == '__main__':
    import sys
    loader = DataLoader(
        sys.argv[1],
        sys.argv[2],
        (sys.argv[3], sys.argv[4]),
        batch_size=128
    )

    print(len(loader.src.vocab))
    print(len(loader.tgt.vocab))

    for batch_index, batch in enumerate(loader.train_iter):
        print(batch.src)
        print(batch.tgt)

        if batch_index > 1:
            break
