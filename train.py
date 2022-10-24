from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from plot import plot_scores


def get_filename(model_args):
    filename = ""
    for key, value in model_args.items():
        filename += f"_{key}_{value}"
    
    return filename


def train_transformer_decoder(device, model, train_dl, test_dl, num_epochs, loss_fn, score_fn, optimizer, eval_name):
    results = {
        'train_scores': [],
        'test_scores': [],
        'losses': [],
    }
    model = model.to(device)
    for e in tqdm(range(num_epochs)):
        model.train()
        avg_loss = 0
        count = 0
        for i, (input, window_len, class_idx) in enumerate(train_dl):
            window_len = window_len[0].item()
            class_idx = class_idx[0].item()
            src = input[:, :window_len, :]
            trg = input[:, 1:1 + window_len, :]
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src)
            loss = loss_fn(out, trg)
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 50 == 0:
                print(f'\nloss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        results['losses'].append(avg_loss)
        
        model.eval()
        train_score = score_fn(model, train_dl, device, eval_name)
        test_score = score_fn(model, test_dl, device, eval_name)
        results['train_scores'].append(train_score.cpu())
        results['test_scores'].append(test_score.cpu())
        print(f"\nEpoch {e + 1} - Train score {train_score} - Test score {test_score}")
    
    return model, results


def train_transformer(device, model, train_dl, test_dl, num_epochs, loss_fn, score_fn, optimizer, eval_name):
    results = {
        'train_scores': [],
        'test_scores': [],
        'losses': [],
    }
    model = model.to(device)
    for e in tqdm(range(num_epochs)):        
        model.train()
        avg_loss = 0
        count = 0
        for i, (input, window_len, class_idx) in enumerate(train_dl):
            window_len = window_len[0].item()
            class_idx = class_idx[0].item()
            _, n, _ = input.shape
            forecast_len = (n - window_len + 1) // 2
            src = input[:, :window_len, :]
            trg = input[:, window_len - 1:window_len - 1 + forecast_len, :]
            trg_y = input[:, window_len:window_len + forecast_len, :]
            src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
            optimizer.zero_grad()
            out = model(src, trg)
            loss = loss_fn(out, trg_y)
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 50 == 0:
                print(f'\nloss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        results['losses'].append(avg_loss)
        
        model.eval()
        train_score = score_fn(model, train_dl, device, eval_name)
        test_score = score_fn(model, test_dl, device, eval_name)
        results['train_scores'].append(train_score.cpu())
        results['test_scores'].append(test_score.cpu())
        print(f"\nEpoch {e + 1} - Train score {train_score} - Test score {test_score}")
    
    return model, results


def train_lstm(device, model, train_dl, test_dl, num_epochs, loss_fn, score_fn, optimizer, eval_name):
    results = {
        'train_scores': [],
        'test_scores': [],
        'losses': [],
    }
    model = model.to(device)
    for e in tqdm(range(num_epochs)):        
        model.train()
        avg_loss = 0
        count = 0
        for i, (input, window_len, class_idx) in enumerate(train_dl):
            window_len = window_len[0].item()
            class_idx = class_idx[0].item()
            src = input[:, :window_len, :]
            trg = input[:, window_len:window_len + 1, :]
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src)
            loss = loss_fn(out, trg)
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 50 == 0:
                print(f'\nloss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        results['losses'].append(avg_loss)
        
        model.eval()
        train_score = score_fn(model, train_dl, device, eval_name)
        test_score = score_fn(model, test_dl, device, eval_name)
        results['train_scores'].append(train_score.cpu())
        results['test_scores'].append(test_score.cpu())
        print(f"\nEpoch {e + 1} - Train score {train_score} - Test score {test_score}")
    
    return model, results


def train_and_test_model(batch_size, learning_rate, weight_decay, num_epochs, forecast_len, train_dataset, test_dataset, model_cls,
                         loss_fn, optim_cls, train_fn, test_fn, eval_fn, training_results_path, predictions_path, weights_path,
                         model_type, eval_name, model_args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = train_dataset.get_scaler()
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = model_cls(**model_args).to(device)
    
    optimizer = optim_cls(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    model, results = train_fn(device, model, train_dl, test_dl, num_epochs, loss_fn, eval_fn, optimizer, eval_name)
    score_name = eval_name.upper()
    loss_name = str(loss_fn).lower()
    if "mse" in loss_name:
        loss_name = "MSE"
    elif "l1" in loss_name:
        loss_name = "MAE"
    else:
        loss_name = ""
    
    filename = get_filename(model_args)
    torch.save(model.state_dict(), f"{weights_path}{model_type}/weights_{filename}.pth")
    
    plot_scores(results['train_scores'], results['test_scores'], results['losses'], loss_name, score_name,
                training_results_path + model_type + f"/training_results_{filename}.png")
    
    mae, mape = test_fn(device, model, test_dl, forecast_len, scaler,
                        save_path=predictions_path + model_type + f"/predictions_{filename}")
    # mae_train, mape_train = test_fn(device, model, train_dl, forecast_len, scaler,
    #                                 save_path=predictions_path + model_type + f"/train/predictions_{filename}")
    
    final_loss = results['losses'][-1]
    final_train_score = results['train_scores'][-1]
    final_test_score = results['test_scores'][-1]
    with open(training_results_path + model_type + f"/training_results_{filename}.txt", "w") as file:
        file.write(f"Final loss ({loss_name}) value\t{final_loss}\n")
        file.write(f"Final train score ({score_name}) value\t{final_train_score}\n")
        file.write(f"Final test score ({score_name}) value\t{final_test_score}\n")
    
    with open(predictions_path + model_type + f"/prediction_results_{filename}.txt", "w") as file:
        file.write(f"MAE score on target\t{mae}\n")
        file.write(f"MAPE score on target\t{mape}\n")
    # with open(predictions_path + model_type + f"/train/prediction_results_{filename}.txt", "w") as file:
    #     file.write(f"MAE score on target\t{mae_train}\n")
    #     file.write(f"MAPE score on target\t{mape_train}\n")
