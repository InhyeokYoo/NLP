import torchtext
from torchtext.data import NestedField
import math

class BPTTIterator(torchtext.data.BPTTIterator):
    """
    Reference: https://github.com/pytorch/text/issues/444#issuecomment-496907298
    custom iterator for hybrid char-word
    """
    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        data = TEXT.pad([text]) # new
        data = TEXT.numericalize(
            data, device=self.device) # data = TEXT.numericalize([text], device=self.device)
        
        # new line start
        size = list(data.size())
        size[0] = self.batch_size
        size[1] = -1
        
        data = data.view(*size).transpose(0, 1).contiguous()
        dataset = torchtext.data.Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(data) - i - 1)
                batch_text = data[i:i + seq_len]
                batch_target = data[i + 1:i + 1 + seq_len]
                if TEXT.batch_first:
                    batch_text = batch_text.transpose(0, 1).contiguous()
                    batch_target = batch_target.transpose(0, 1).contiguous()
                yield torchtext.data.Batch.fromvars(
                    dataset, self.batch_size,
                    text=batch_text,
                    target=batch_target)
            if not self.repeat:
                return

def gen_bptt_iter(dataset, batch_size, bptt_len, device):
    # dataset: tuple of dataset
    for batch_word, batch_char in zip(
            BPTTIterator(dataset[0], batch_size, bptt_len, device=device),
            BPTTIterator(dataset[1], batch_size, bptt_len, device=device),
    ):
        yield batch_word.text, batch_char.text, batch_word.target, batch_char.target

def gen_language_model_corpus(dataset_cls: torchtext.datasets.LanguageModelingDataset):
    field_char = NestedField(Field(
        pad_token=PAD_WORD,
        tokenize=list,
        init_token=SOS_WORD,
        eos_token=EOS_WORD,
        batch_first=True),
        pad_token=PAD_WORD,
    )

    field_word = Field(batch_first=True)
    dataset_char = dataset_cls.splits(field_char)
    dataset_word = dataset_cls.splits(field_word)
    field_char.build_vocab(dataset_char[0])
    field_word.build_vocab(dataset_word[0])
    return [_ for _ in zip(dataset_word, dataset_char)], field_word, field_char

# How to use:
if __name__ == "__main__":
    from torchtext.datasets import WikiText2
    from torchtext.data import Field

    # FINAL
    PAD_WORD = '<pad>'
    SOS_WORD = '<sow>'
    EOS_WORD = '<eow>'

    datasets, field_word, field_char = gen_language_model_corpus(WikiText2)
    train_data, valid_data, test_data = datasets

    '''
    datasets:
    # The list contains train, valid and test dataset. Each tuple in the list is the dataset of word tokens and the dataset of character tokens
    [(<torchtext.datasets.language_modeling.WikiText2 at 0x7f8993b23b70>,
      <torchtext.datasets.language_modeling.WikiText2 at 0x7f89b65803c8>),
     (<torchtext.datasets.language_modeling.WikiText2 at 0x7f8991d53908>,
      <torchtext.datasets.language_modeling.WikiText2 at 0x7f8995cc70f0>),
     (<torchtext.datasets.language_modeling.WikiText2 at 0x7f898bb10080>,
      <torchtext.datasets.language_modeling.WikiText2 at 0x7f8995c0af98>)]

    train_data[0].__dict__ # idx 0 stands for word, 1 stands for char
    {'examples': [<torchtext.data.example.Example at 0x7f898bb10048>],
     'fields': {'text': <torchtext.data.field.Field at 0x7f89b6580d30>}}

    # you can access field attr of the dataset
    train_data[1].fields['text'].vocab.stoi
    # or
    field_char.vocab.stoi 

    # and make Iterator for train a model by using `gen_bptt_iter`
    # **NOTE**: train_iter is a generator, so it will be consumed when you train the model.
    train_iter = gen_bptt_iter(train_data, 32, 30, 'cpu')

    # word_text, char_text: word, char train text data respectively
    # word_target, char_taget: word, char train label data respectively (lagging t+1)
    for word_text, char_text, word_target, char_taget in train_iter: 
        print(f"Words - {word_text.size()} \n {' '.join([field_word.vocab.itos[word.item()] for word in word_text[0, :]])}")
        chars = []
        for i in range(30):
            temp = [''.join([field_char.vocab.itos[word.item()] for word in char_text[0, i, :] if field_char.vocab.itos[word.item()] not in tokens])]
            chars.extend(temp)
        print(f"Words - {char_text.size()} \n {' '.join(chars)}")

        print("Targets:")
        print(f"Words - {word_target.size()} \n {' '.join([field_word.vocab.itos[word.item()] for word in word_target[0, :]])}")
        chars = []
        for i in range(30):
            temp = [''.join([field_char.vocab.itos[word.item()] for word in char_taget[0, i, :] if field_char.vocab.itos[word.item()] not in tokens])]
            chars.extend(temp)
        print(f"Words - {char_text.size()} \n {' '.join(chars)}")

    
    '''