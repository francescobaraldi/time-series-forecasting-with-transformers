import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_scores(train_scores, test_scores, losses, save_path=None):
    legend = ['Train', 'Test']
    xlabel = 'Epoch'
    num_epochs = len(train_scores)

    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), losses, 'r')
    plt.title('Loss')
    plt.xlabel(xlabel)
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_scores, 'g', \
        range(1, num_epochs + 1), test_scores, 'b')
    plt.title('MAE')
    plt.xlabel(xlabel)
    plt.ylabel('MAE')
    plt.legend(legend)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + f"training_results_{num_epochs}_epochs.png")
    

def plot_predictions(device, model, dl, scaler, save_path=None):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (src, trg, class_idx) in enumerate(dl):
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            plt.figure(figsize=(15,6))
            src_rec = scaler.inverse_transform(src[0, :, :].cpu())
            trg_rec = scaler.inverse_transform(trg[0, :, :].cpu() + torch.zeros((7, 10)))
            out_rec = scaler.inverse_transform(out[0, :, :].cpu() + torch.zeros((7, 10)))
            src_rec = src_rec[:, class_idx].tolist()
            trg_rec = trg_rec[:, 0].tolist()
            out_rec = out_rec[:, 0].tolist()
            plt.plot(np.arange(1, len(src_rec) + 1), src_rec, '-', color='green', label='Source', linewidth=2)
            plt.plot(np.arange(1, len(trg_rec) + 1), trg_rec, '-', color='blue', label='Target', linewidth=2)
            plt.plot(np.arange(1, len(out_rec) + 1), out_rec, '--', color='red', label='Prediction', linewidth=2)
            plt.xlabel("Days")
            plt.ylabel("Closing price ($)")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            if save_path is None:
                plt.show()
            else:
                plt.savefig(save_path + f"prediction_{i}.png")
            
