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
        plt.savefig(save_path)
        

def plot_predictions(src, trg, predictions, scaler, forecast_len, class_idx, i, save_path=None):
    _, window_len, input_size = src.shape
    src_rec = scaler.inverse_transform(src[0, :, :])
    trg_rec = scaler.inverse_transform(trg[0, :, :])
    predictions_rec = scaler.inverse_transform(predictions[0, :, :] + torch.zeros((window_len, input_size)))
    src_rec = src_rec[:, class_idx].tolist()
    trg_rec = trg_rec[:, class_idx].tolist()
    predictions_rec = predictions_rec[:, 0].tolist()
    plt.figure(figsize=(15,6))
    plt.plot(np.arange(1, window_len + 1), src_rec, '-', color='green', label='Source', linewidth=4, alpha=0.5)
    plt.plot(np.arange(1 + forecast_len, window_len + forecast_len + 1), trg_rec, '-', color='blue', label='Target', linewidth=1)
    plt.plot(np.arange(1 + forecast_len,  window_len + forecast_len + 1), predictions_rec, '--', color='red', label='Prediction', linewidth=1)
    plt.xlabel("Days")
    plt.ylabel("Closing price ($)")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + f"_{i}.png")


def plot_predictions2(src, trg, predictions, scaler, forecast_len, class_idx, i, save_path=None):
    _, window_len, input_size = src.shape
    src_rec = scaler.inverse_transform(src[0, :, :])
    trg_rec = scaler.inverse_transform(trg[0, :, :])
    predictions_rec = scaler.inverse_transform(predictions[0, :, :] + torch.zeros((predictions.shape[1], input_size)))
    src_rec = src_rec[:, class_idx].tolist()
    trg_rec = trg_rec[:, class_idx].tolist()
    predictions_rec = predictions_rec[:, 0].tolist()
    plt.figure(figsize=(15,6))
    plt.plot(np.arange(1, window_len + 1), src_rec, '-', color='green', label='Source', linewidth=4, alpha=0.5)
    plt.plot(np.arange(1 + window_len, 1 + window_len + forecast_len), trg_rec, '-', color='blue', label='Target', linewidth=1)
    plt.plot(np.arange(2, 1 + window_len + forecast_len), predictions_rec, '--', color='red', label='Prediction', linewidth=1)
    plt.xlabel("Days")
    plt.ylabel("Closing price ($)")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + f"prediction_{i}.png")
