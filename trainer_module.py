import torch
from torch import optim
import numpy as np
import time
import copy
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

def training(
    model,
    train_X,
    train_Y,
    val_X,
    val_Y,
    max_epochs=10000,
    patience=1000,
    batch_size = 256,
    lr=1e-3,
    y_scaler = None,
    device = 'cpu',
    LAMBDA = 1,
    verbose = False
):
    # device mount
    model = model.to(device)
    train_X1 = train_X.to(device)
    train_Y1 = train_Y.to(device)
    val_X1 = val_X.to(device)
    val_Y1 = val_Y.to(device)
    train_dataset = TensorDataset(train_X1, train_Y1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # init
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = 1e15
    best_epoch = 0
    best_model = None
    early_stopping = 0
    preepoch_time = time.time()
    
    for epoch in range(max_epochs):
        # train
        model.train()
        for batch_X, batch_Y in train_loader:
            # device mount
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            # forward
            w_hat, w_tilde, _, y_tilde = model(batch_X)
            
            # augmented covariates [x, 1]
            new_batch_X = torch.cat([batch_X, torch.ones((batch_X.shape[0],1), device=device).type(torch.float32)], axis=1)
            
            # loss
            alignment_loss = torch.mean(((new_batch_X * w_tilde).sum(1).reshape(-1,1) - (new_batch_X * w_hat).sum(1).reshape(-1,1)) ** 2)
            prediction_fit_loss = torch.mean((y_tilde - batch_Y) ** 2)
            total_loss = prediction_fit_loss + (1+LAMBDA) * alignment_loss

            # parameter update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # validation
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                # forward
                _, _, _, y_tilde_val = model(val_X1) 
                if y_scaler:
                    output_loss_val = np.mean((y_scaler.inverse_transform(y_tilde_val.cpu().detach().numpy()) - y_scaler.inverse_transform(val_Y1.cpu()))**2)
                else:
                    output_loss_val = torch.mean((y_tilde_val - val_Y1)**2)
                   
                # save best model
                if best_loss > output_loss_val:
                    best_loss = output_loss_val
                    torch.save(model, "best_model.pt")
                    best_model = copy.deepcopy(model)

                    # reset patient
                    early_stopping = 0
                    best_epoch = epoch+1
                else:
                    early_stopping += 1

        # verbose option
        if verbose and (epoch % 100 == 99):
            model.eval()
            w_hat_train, _, _, y_tilde_train = model(train_X1)
            if y_scaler:
                codebook_loss_train = np.mean((y_scaler.inverse_transform(y_tilde_train.cpu().detach().numpy()) - y_scaler.inverse_transform(train_Y1.cpu()))**2)
                new_batch_X = torch.cat([train_X1, torch.ones((train_X1.shape[0],1), device=device).type(torch.float32)], axis=1)
                proxy_loss_train = np.mean((y_scaler.inverse_transform(((new_batch_X*w_hat_train).sum(1).reshape(-1,1)).cpu().detach().numpy()) - y_scaler.inverse_transform(train_Y1.cpu()))**2)                
            else:
                codebook_loss_train = np.mean((y_tilde_train.cpu().detach().numpy() - train_Y1.cpu())**2)
                new_batch_X = torch.cat([train_X1, torch.ones((train_X1.shape[0],1), device=device).type(torch.float32)], axis=1)
                proxy_loss_train = np.mean((((new_batch_X*w_hat_train).sum(1).reshape(-1,1).cpu().detach().numpy()) - train_Y1.cpu())**2)
            
            print(f"[Epoch {epoch+1} ({time.time()-preepoch_time}s)] TRAIN RMSE: Codebook({round(codebook_loss_train**0.5,3)})|Proxy({round(proxy_loss_train**0.5,3)}, [Best Epoch {best_epoch}] Best VAL RMSE: Codebook({round(best_loss**0.5,3)})")  
            preepoch_time = time.time()
        
        if early_stopping > patience:
            print(f"Training End... Epoch {epoch+1} (Early Stopping at {best_epoch} with VAL RMSE {round(best_loss**0.5,3)})")
            break
            
    # return
    time.sleep(1)
    try:
        best_model = torch.load("best_model.pt")
    except:
        best_model = best_model
        
    return best_model
