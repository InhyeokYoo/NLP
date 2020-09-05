import matplotlib.pyplot as plt

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