from tqdm import tqdm


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
        for i, (src, trg, _) in enumerate(train_dl):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            out = model(src)
            loss = loss_fn(out, trg)
            avg_loss += loss.cpu().detach().numpy().item()
            if i % 10 == 0:
                print(f'loss {loss.cpu().item():.6f}')
            loss.backward()
            optimizer.step()
            count += 1
        avg_loss /= count
        results['losses'].append(avg_loss)
    
    return model, results
