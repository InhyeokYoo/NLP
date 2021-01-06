import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sentencepiece as spm
import time
import trainer
import data_loader
import dataset
import model

if __name__ == "__main__":
    # Args:
    args = {
        'dataset': {
            "train_input_file": "/content/drive/MyDrive/Colab-Notebooks/datasets/BookCorpus_train.txt",
            "test_input_file": "/content/drive/MyDrive/Colab-Notebooks/datasets/BookCorpus_test.txt",
            "val_input_file": "/content/drive/MyDrive/Colab-Notebooks/datasets/BookCorpus_val.txt"
            },
        'wordpiece': {
            "vocab_size": 30000,
            "prefix": 'bookcorpus_spm',
            'user_defined_symbols': '[PAD],[CLS],[SEP],[MASK]',
            'model_type': 'bpe',
            'character_coverage': 1.0, # default
        },
        'model': {
            "seq_len":512,
            "mask_frac":0.15,
        },
        'train': {
            "batch_size": 24,
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "weight_decay": 0.01,
            "weight_decay": 0.01,
            "device": torch.device('cuda' if torch.cuda.is_available() == True else 'cpu'),
        }
    }

    # Load dataset:
    train_data, val_data, test_data = dataset.load_dataset(args["dataset"]['train_input_file'], 
                                                            args["dataset"]["val_input_file"],
                                                            args["dataset"]['test_input_file'])
    parameter = '--input={} --model_prefix={} --vocab_size={} --user_defined_symbols={} --model_type={} --character_coverage={}'

    cmd = parameter.format(
        args['dataset']['train_input_file'], args['wordpiece']['prefix'], 
        args['wordpiece']['vocab_size'], args['wordpiece']['user_defined_symbols'], 
        args['wordpiece']['model_type'], args['wordpiece']['character_coverage'])

    spm.SentencePieceTrainer.Train(cmd)
    sp = spm.SentencePieceProcessor()
    sp.Load(f"/content/drive/MyDrive/Colab-Notebooks/datasets/{args['wordpiece']['prefix']}.model")

    args['sep_id'] = sp.piece_to_id('[SEP]')        
    args['cls_id'] = sp.piece_to_id('[CLS]')
    args['mask_id'] = sp.piece_to_id('[MASK]')
    args['pad_id'] = sp.piece_to_id('[PAD]')

    train_data = dataset.BERTLanguageModelingDataset(train_data, sp)
    val_data = dataset.BERTLanguageModelingDataset(val_data, sp)
    test_data = dataset.BERTLanguageModelingDataset(test_data, sp)

    train_loader = DataLoader(train_data, batch_size=args["batch_size"], shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args["batch_size"], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args["batch_size"], shuffle=False)
    
    # Create model:
    bert = model.BertModel()
    mlm_head = bert

    mlm_optimizer = optim.Adam(mlm_head.parameters(), lr=1e-4, betas=args["betas"], weight_decay=args["weight_decay"])
    nsp_optimizer = optim.Adam(nsp_head.parameters(), lr=1e-4, betas=[0.9, 0.999], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()


    N_EPOCHS = 10

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, N_EPOCHS+1):
        start_time = time.time()
        mlm_loss, nsp_loss = train(mlm_head, nsp_head, dataloader, mlm_optimizer, nsp_optimizer, criterion, 1)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {mlm_loss:.3f} | Train PPL: {math.exp(nsp_loss):7.3f}')