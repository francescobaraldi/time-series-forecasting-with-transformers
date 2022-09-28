import torch
import matplotlib.pyplot as plt


def plot_scores(train_maes, test_maes, losses):
    legend = ['Train', 'Test']
    xlabel = 'Epoch'
    n_epochs = len(train_maes)

    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs + 1), losses, 'r')
    plt.title('Loss')
    plt.xlabel(xlabel)
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs + 1), train_maes, 'g', \
        range(1, n_epochs + 1), test_maes, 'b')
    plt.title('MAE')
    plt.xlabel(xlabel)
    plt.ylabel('MAE')
    plt.legend(legend)
    plt.show()
    

def plot_predictions(device, model, dl, scaler, save_path):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (src, trg, class_idx) in enumerate(dl):
            src, trg = src.to(device), trg.to(device)
            out = model(src)
            plt.figure(figsize=(15,6))
            src_rec = scaler.inverse_transform(src[0, :, :])
            trg_rec = scaler.inverse_transform(trg[0, :, :] + torch.zeros((7, 10)))
            out_rec = scaler.inverse_transform(out[0, :, :] + torch.zeros((7, 10)))
            src_rec = src_rec[:, class_idx].tolist()
            trg_rec = trg_rec[:, 0].tolist()
            out_rec = out_rec[:, 0].tolist()
            plt.plot(src_rec, '-', color = 'green', label = 'Src', linewidth=2)
            plt.plot(trg_rec, '-', color = 'blue', label = 'Target', linewidth=2)
            plt.plot(out_rec, '--', color = 'red', label = 'Prediction', linewidth=2)
            plt.savefig(save_path + f"prediction_{i}.png")
            
