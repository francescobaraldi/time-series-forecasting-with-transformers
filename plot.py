import numpy as np
import matplotlib.pyplot as plt


def plot_scores(train_scores, test_scores, losses, loss_name="", score_name="", save_path=None):
    legend = ['Train', 'Test']
    xlabel = 'Epoch'
    num_epochs = len(train_scores)
    

    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), losses, color='r')
    plt.title('Loss evolution during training')
    plt.xlabel(xlabel)
    plt.ylabel(f'Loss ({loss_name})')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_scores, 'g', \
        range(1, num_epochs + 1), test_scores, 'b')
    plt.title('Score evolution during training')
    plt.xlabel(xlabel)
    plt.ylabel("Score" if score_name == "" else score_name)
    plt.legend(legend)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_predictions(src, trg, predictions, scaler, forecast_len, class_idx, i, save_path=None):
    _, window_len, _ = src.shape
    src_rec = scaler.inverse_transform(src[0, :, :])
    trg_rec = scaler.inverse_transform(trg[0, :, :])
    predictions_rec = scaler.inverse_transform(predictions[0, :, :])
    src_rec = src_rec[:, class_idx].tolist()
    trg_rec = trg_rec[:, class_idx].tolist()
    predictions_rec = predictions_rec[:, class_idx].tolist()
    
    plt.figure(figsize=(15,6))
    plt.plot(np.arange(1, window_len + 1), src_rec, '-', color='green', label='Source', linewidth=1)
    plt.plot(np.arange(1 + window_len, 1 + window_len + forecast_len), trg_rec, '-', color='blue', label='Target', linewidth=1)
    plt.plot(np.arange(1 + window_len, 1 + window_len + forecast_len), predictions_rec, '--', color='red', label='Prediction', linewidth=1)
    plt.xlabel("Days")
    plt.ylabel("Closing price ($)")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + f"_prediction_{i}.png")


def plot_inference(src, predictions, scaler, forecast_len, class_idx, save_path=None):
    window_len, _ = src.shape
    src_rec = scaler.inverse_transform(src)
    predictions_rec = scaler.inverse_transform(predictions)
    src_rec = src_rec[:, class_idx].tolist()
    predictions_rec = predictions_rec[:, class_idx].tolist()
    
    plt.figure(figsize=(15,6))
    plt.plot(np.arange(1, window_len + 1), src_rec, '-', color='green', label='Source', linewidth=1)
    plt.plot(np.arange(1 + window_len, 1 + window_len + forecast_len), predictions_rec, '--', color='red', label='Prediction', linewidth=1)
    plt.xlabel("Days")
    plt.ylabel("Closing price ($)")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
