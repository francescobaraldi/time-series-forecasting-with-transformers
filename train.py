from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from plot import plot_scores
from test import test


def train_model(device, model, train_dl, test_dl, num_epochs, loss_fn, score_fn, optimizer):
    results = {
        'train_scores': [],
        'test_scores': [],
        'losses': [],
    }
    model = model.to(device)
    for e in tqdm(range(num_epochs)):
        model.eval()
        train_score = score_fn(model, train_dl, device)
        test_score = score_fn(model, test_dl, device)
        results['train_scores'].append(train_score.cpu())
        results['test_scores'].append(test_score.cpu())
        print(f"Epoch {e} - Train score {train_score} - Test score {test_score}")
        
        model.train()
        avg_loss = 0
        count = 0
        for i, (src, trg, class_idx) in enumerate(train_dl):
            class_idx = class_idx[0].item()
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src)
            loss = loss_fn(out, trg[:, :, class_idx].unsqueeze(-1))
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 10 == 0:
                print(f'loss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        results['losses'].append(avg_loss)
    
    return model, results


def train_model2(device, model, train_dl, test_dl, num_epochs, loss_fn, score_fn, optimizer):
    results = {
        'train_scores': [],
        'test_scores': [],
        'losses': [],
    }
    model = model.to(device)
    for e in tqdm(range(num_epochs)):
        model.eval()
        train_score = score_fn(model, train_dl, device)
        test_score = score_fn(model, test_dl, device)
        results['train_scores'].append(train_score.cpu())
        results['test_scores'].append(test_score.cpu())
        print(f"Epoch {e} - Train score {train_score} - Test score {test_score}")
        
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
            if i % 100 == 0:
                print(f'loss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        results['losses'].append(avg_loss)
    
    return model, results


def train_model_std(device, model, train_dl, test_dl, num_epochs, loss_fn, score_fn, optimizer):
    results = {
        'train_scores': [],
        'test_scores': [],
        'losses': [],
    }
    model = model.to(device)
    for e in tqdm(range(num_epochs)):
        model.eval()
        train_score = score_fn(model, train_dl, device)
        test_score = score_fn(model, test_dl, device)
        results['train_scores'].append(train_score.cpu())
        results['test_scores'].append(test_score.cpu())
        print(f"Epoch {e} - Train score {train_score} - Test score {test_score}")
        
        model.train()
        avg_loss = 0
        count = 0
        for i, (input, trg_y, class_idx) in enumerate(train_dl):
            class_idx = class_idx[0].item()
            src = input[:, :-1, :]
            trg = input[:, 1:, :]
            src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
            optimizer.zero_grad()
            out = model(src, trg)
            loss = loss_fn(out, trg_y[:, :, class_idx].unsqueeze(-1))
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 100 == 0:
                print(f'loss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        results['losses'].append(avg_loss)
    
    return model, results


def train_and_test_model(batch_size, learning_rate, num_epochs, window_len, forecast_len, input_size, output_size, num_layer,
                         dropout, train_dataset, test_dataset, model_cls, loss_fn, optim_cls, train_fn, eval_fn,
                         training_results_path, predictions_path, model_type, d_model=None):
    
    if d_model is None:
        d_model = input_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = train_dataset.get_scaler()
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    model = model_cls(seq_len=window_len, num_layer=num_layer, input_size=input_size, output_size=output_size, d_model=d_model, num_heads=d_model, feedforward_dim=64, dropout=dropout).to(device)
    optimizer = optim_cls(model.parameters(), lr=learning_rate)
    
    model, results = train_fn(device, model, train_dl, test_dl, num_epochs, loss_fn, eval_fn, optimizer)
    
    plot_scores(results['train_scores'], results['test_scores'], results['losses'], training_results_path + model_type +
                f"/training_results_batch_size_{batch_size}_learning_rate_{learning_rate}_num_epochs_{num_epochs}_num_layer_{num_layer}_d_model_{d_model}_dropout_{dropout}.png")
    test(device, model, test_dl, forecast_len, scaler, save_path=predictions_path + model_type +
         f"/predictions_batch_size_{batch_size}_learning_rate_{learning_rate}_num_epochs_{num_epochs}_num_layer_{num_layer}_d_model_{d_model}_dropout_{dropout}")
