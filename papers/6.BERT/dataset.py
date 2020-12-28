def load_dataset(train_file, val_file, test_file):
    '''
    load dataset for BERT and return train, val, test dataset. 
    Dataset is a text file. Each line contains one sentence.
    '''
    def read_text(file):
        dataset = []

        with open(file, 'r') as f:
            for data in f:
                dataset.append(data.strip())

        return dataset
    
    train = read_text(train_file)
    test = read_text(test_file)
    val = read_text(val_file)

    return train, val, test