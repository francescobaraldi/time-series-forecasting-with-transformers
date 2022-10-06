from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from plot import plot_scores


def get_filename(model_args):
    filename = ""
    for key, value in model_args.items():
        filename += f"_{key}_{value}"
    
    return filename


def train_model_singlestep(device, model, train_dl, test_dl, num_epochs, loss_fn, score_fn, optimizer, scaler=None):
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
        for i, (input, _, class_idx) in enumerate(train_dl):
            class_idx = class_idx[0].item()
            src = input[:, :-1, :]
            trg = input[:, 1:, :]
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src)
            loss = loss_fn(out, trg[:, :, class_idx].unsqueeze(-1))
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 50 == 0:
                print(f'\nloss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        results['losses'].append(avg_loss)
        
        model.eval()
        train_score, score_name = score_fn(model, train_dl, device, scaler)
        test_score, _ = score_fn(model, test_dl, device, scaler)
        results['train_scores'].append(train_score.cpu())
        results['test_scores'].append(test_score.cpu())
        print(f"\nEpoch {e} - Train score {train_score} - Test score {test_score}")
    
    return model, results, score_name


def train_model_multistep(device, model, train_dl, test_dl, num_epochs, loss_fn, score_fn, optimizer, scaler=None):
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
        for i, (input, trg, class_idx) in enumerate(train_dl):
            class_idx = class_idx[0].item()
            src = input[:, :-1, :]
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src)
            loss = loss_fn(out, trg[:, :, class_idx].unsqueeze(-1))
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 50 == 0:
                print(f'\nloss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        results['losses'].append(avg_loss)
        
        model.eval()
        train_score, score_name = score_fn(model, train_dl, device, scaler)
        test_score, _ = score_fn(model, test_dl, device, scaler)
        results['train_scores'].append(train_score.cpu())
        results['test_scores'].append(test_score.cpu())
        print(f"\nEpoch {e} - Train score {train_score} - Test score {test_score}")
    
    return model, results, score_name


def train_and_test_model(batch_size, learning_rate, num_epochs, window_len, forecast_len, input_size, output_size, d_model, num_heads,
                         num_layers, dropout, feedforward_dim, positional_encoding, train_dataset, test_dataset, model_cls, loss_fn,
                         optim_cls, train_fn, test_fn, eval_fn, training_results_path, predictions_path, model_type, step_type):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = train_dataset.get_scaler()
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = model_cls(seq_len=window_len, num_layers=num_layers, input_size=input_size, output_size=output_size, d_model=d_model,
                      num_heads=num_heads, feedforward_dim=feedforward_dim, dropout=dropout,
                      positional_encoding=positional_encoding).to(device)
    
    optimizer = optim_cls(model.parameters(), lr=learning_rate)
    
    model, results, score_name = train_fn(device, model, train_dl, test_dl, num_epochs, loss_fn, eval_fn, optimizer, scaler)
    loss_name = str(loss_fn).lower()
    if "mse" in loss_name:
        loss_name = "MSE"
    elif "l1" in loss_name:
        loss_name = "MAE"
    else:
        loss_name = ""
    
    filename = f"_num_epochs_{num_epochs}_num_layers_{num_layers}_d_model_{d_model}_num_heads_{num_heads}_dropout_{dropout}_feedforward_dim_{feedforward_dim}_positional_encoding_{positional_encoding}"
    
    plot_scores(results['train_scores'], results['test_scores'], results['losses'], loss_name, score_name,
                training_results_path + model_type + "/" + step_type + f"/training_results_{filename}.png")
    
    src_mae, src_mape, trg_mae, trg_mape = test_fn(device, model, test_dl, forecast_len, scaler,
                                                save_path=predictions_path + model_type + "/" + step_type + f"/predictions_{filename}")
    
    best_loss = results['losses'][-1]
    best_train_score = results['train_scores'][-1]
    best_test_score = results['test_scores'][-1]
    with open(training_results_path + model_type + f"/{step_type}/training_results_{filename}.txt", "w") as file:
        file.write(f"Final loss ({loss_name}) value\t{best_loss}\n")
        file.write(f"Final train score ({score_name}) value\t{best_train_score}\n")
        file.write(f"Final test score ({score_name}) value\t{best_test_score}\n")
    
    with open(predictions_path + model_type + f"/{step_type}/prediction_results_{filename}.txt", "w") as file:
        file.write(f"MAE score on source\t{src_mae}\n")
        file.write(f"MAPE score on source\t{src_mape}\n")
        file.write(f"MAE score on target\t{trg_mae}\n")
        file.write(f"MAPE score on target\t{trg_mape}\n")


def train_and_test_model2(batch_size, learning_rate, num_epochs, forecast_len, train_dataset,
                          test_dataset, model_cls, loss_fn, optim_cls, train_fn, test_fn, eval_fn, training_results_path,
                          predictions_path, model_type, step_type, model_args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = train_dataset.get_scaler()
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = model_cls(**model_args).to(device)
    
    optimizer = optim_cls(model.parameters(), lr=learning_rate)
    
    model, results, score_name = train_fn(device, model, train_dl, test_dl, num_epochs, loss_fn, eval_fn, optimizer, scaler)
    loss_name = str(loss_fn).lower()
    if "mse" in loss_name:
        loss_name = "MSE"
    elif "l1" in loss_name:
        loss_name = "MAE"
    else:
        loss_name = ""
    
    filename = get_filename(model_args)
    
    plot_scores(results['train_scores'], results['test_scores'], results['losses'], loss_name, score_name,
                training_results_path + model_type + "/" + step_type + f"/training_results_{filename}.png")
    
    src_mae, src_mape, trg_mae, trg_mape = test_fn(device, model, test_dl, forecast_len, scaler,
                                                save_path=predictions_path + model_type + "/" + step_type + f"/predictions_{filename}")
    
    best_loss = results['losses'][-1]
    best_train_score = results['train_scores'][-1]
    best_test_score = results['test_scores'][-1]
    with open(training_results_path + model_type + f"/{step_type}/training_results_{filename}.txt", "w") as file:
        file.write(f"Final loss ({loss_name}) value\t{best_loss}\n")
        file.write(f"Final train score ({score_name}) value\t{best_train_score}\n")
        file.write(f"Final test score ({score_name}) value\t{best_test_score}\n")
    
    with open(predictions_path + model_type + f"/{step_type}/prediction_results_{filename}.txt", "w") as file:
        file.write(f"MAE score on source\t{src_mae}\n")
        file.write(f"MAPE score on source\t{src_mape}\n")
        file.write(f"MAE score on target\t{trg_mae}\n")
        file.write(f"MAPE score on target\t{trg_mape}\n")
