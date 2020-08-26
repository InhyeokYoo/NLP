'''
Reference: https://github.com/kh-kim/simple-nmt/blob/master/simple_nmt/data_loader.py
'''

from torchtext import data, datasets

class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        if not path.endswith('.'):
            path += '.'

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length < max(len(src_line.split()), len(trg_line.split())):
                    continue
                if src_line != '' and trg_line != '':
                    examples += [data.Example.fromlist([src_line, trg_line], fields)]

        super().__init__(examples, fields, **kwargs)


if __name__ == '__main__':
    train_fn = None
    valid_fn = None
    train = TranslationDataset(
        path=train_fn, exts=exts,
        fields=[('src', self.src), ('tgt', self.tgt)],
        max_length=max_length)

    valid = TranslationDataset(
        path=valid_fn, 
        exts=exts,
        fields=[('src', self.src), ('tgt', self.tgt)],
        max_length=max_length,)