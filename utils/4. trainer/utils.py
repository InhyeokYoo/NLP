import matplotlib.pyplot as plt
import math, time
from torchtext.data.metrics import bleu_score

def plot_losses(train_losses, test_losses):
    '''
    Plot both train_losses and test_losses
    param:
        train_losses: a list that containing train losses of the model
        test_loses: a list that containing test losses of the model
    '''
    plt.plot(train_losses, 'r--', label="Train")
    plt.plot(test_losses, 'b', label="Test")
    plt.legend()

    plt.show()

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def show_train_info(epoch, start_time, end_time, train_loss, valid_loss, **kwargs):
    # kwargs for bleu_score:
    # https://pytorch.org/text/data_metrics.html?highlight=bleu#torchtext.data.metrics.bleu_score
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train BLEU: {bleu_score(**kwargs):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. BLEU: {bleu_score(**kwargs):7.3f}')

def show_evaluate_loss(test_loss, **kwargs):
    print(f'| Test Loss: {test_loss:.3f} | Test BLEU: {bleu_score(**kwargs):7.3f} |')