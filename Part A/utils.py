import torch
import time
from sklearn.metrics import confusion_matrix, accuracy_score

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device
              ):
    
    start_time = time.time()

    train_loss = 0    
    model.to(device)

    for (X, y) in data_loader:
        # send data to GPU
        X, y = X.to(device), y.to(device)
        # X, y = X.to(device), y.type(torch.LongTensor).to(device)
        
        # 1. forward pass
        y_pred = model(X)

        # 2. calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        
        # 3. optimizer zero grad
        optimizer.zero_grad()
        
        # 4. loss backward
        loss.backward()
        
        # 5. optimizer step
        optimizer.step()
    
    train_loss /= len(data_loader)

    end_time = time.time()

    return {"avg_batch_loss": train_loss, "time": (end_time - start_time)* 10**3}

def valid_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               device: torch.device
              ):
    
    # send the model to device
    model.to(device)

    # send the model in eval mode
    model.eval()

    # for confusion matrix and accuracy
    y_true = torch.Tensor([]).to(device)
    y_pred = torch.Tensor([]).to(device)

    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            y_true = torch.cat((y_true, y), dim=0)
            y_pred = torch.cat((y_pred, test_pred.argmax(axis=1)), dim=0)
        
        # send back to cpu
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

        return {"accuracy": accuracy_score(y_true, y_pred), "confusion_matrix": confusion_matrix(y_true, y_pred, normalize="true")}
