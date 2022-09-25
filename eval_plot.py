import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def eval(model, dl, device):
    total_error = 0
    total = 0
    
    with torch.no_grad():
        for seq, trg in dl:
            seq, trg = seq.to(device), trg.to(device)
            seq_mask = torch.ones_like(seq)
            out = model(seq, seq_mask)
            out = out.squeeze(-1)
            trg = trg.squeeze(-1)
            mape = torch.sum(torch.abs((out - trg)))
            total_error += mape.item()
            total += out.size(0)
    
    return total_error / total


def plot_scores(train_maes, test_maes):
    legend = ['Train', 'Test']
    xlabel = 'Epoch'
    n_epochs = len(train_maes)

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs + 1), train_maes, 'g', \
        range(1, n_epochs + 1), test_maes, 'b')
    plt.title('MAE')
    plt.xlabel(xlabel)
    plt.ylabel('MAE')
    plt.legend(legend)
    plt.show()
