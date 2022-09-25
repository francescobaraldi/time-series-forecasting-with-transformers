import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def eval(model, dl, device):
    cum_score = 0
    total = 0
    
    with torch.no_grad():
        for seq, trg in dl:
            seq, trg = seq.to(device), trg.to(device)
            seq_mask = torch.ones_like(seq)
            out = model(seq, seq_mask)
            # out = out.squeeze(-1)
            # trg = trg.squeeze(-1)
            mae = torch.mean(torch.abs((out - trg)))
            cum_score += mae
            total += 1
    
    return cum_score / total


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
