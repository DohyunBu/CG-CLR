# main.py
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_module import load_data
from model_module import CGCLR
from trainer_module import training
import time
from sklearn.model_selection import KFold
import argparse

seed = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    final_result = []
    
    for data_name in args.dataset_name_list:
        result_preformance = []
        full_X, full_Y = load_data(data_name)
        
        # test fold (trainval : test = 4 : 1)
        outer_kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
        for outer_epoch, (trainval_idx, test_idx) in enumerate(outer_kfold.split(range(len(full_X)))):
            trainval_X, test_X, trainval_Y, test_Y = full_X[trainval_idx], full_X[test_idx], full_Y[trainval_idx], full_Y[test_idx]
            
            # validation fold (train : validation = 3 : 1)
            inner_kfold = KFold(n_splits=4, shuffle=True, random_state=seed)
            for inner_epoch, (train_idx, val_idx) in enumerate(inner_kfold.split(range(len(trainval_X)))):
                train_X = trainval_X[train_idx]
                train_Y = trainval_Y[train_idx]
                val_X = trainval_X[val_idx]
                val_Y = trainval_Y[val_idx]
                N_train = train_X.shape[0]
                p_dim = train_X.shape[1]
                if args.num_K != None:
                    if args.num_K <= 0:
                        raise Exception(f"Check num_K variable! {args.num_K} is not positive integer value")
                    num_K = args.num_K
                elif args.coverage == 'small':
                    num_K = N_train//(1*(p_dim+1)) + 1
                elif args.coverage == 'medium':
                    num_K = N_train//(5*(p_dim+1))
                elif args.coverage == 'large':
                    num_K = N_train//(10*(p_dim+1))
                elif args.coverage == 'xlarge':
                    num_K = N_train//(20*(p_dim+1))
                else:
                    raise Exception(f"Check coverage variable! {args.coverage} is not include in ('small','medium','large','xlarge')")
                    
                
                # define model
                model = CGCLR(
                    input_dim=p_dim,
                    expert_num= num_K,
                    output_dim=1,
                    proxy_hidden_shape=args.proxy_hidden_shape,
                    dropout = args.dropout,
                    device = device
                )
                
                # preprocessing
                scaler_x = StandardScaler()
                train_X1 = scaler_x.fit_transform(train_X)
                val_X1 = scaler_x.transform(val_X)
                test_X1 = scaler_x.transform(test_X)
                scaler_y = StandardScaler()
                train_Y1 = scaler_y.fit_transform(train_Y)
                val_Y1 = scaler_y.transform(val_Y)
                test_Y1 = scaler_y.transform(test_Y)
                train_X1 = torch.tensor(np.array(train_X1)).reshape(-1, train_X1.shape[1]).float()
                train_Y1 = torch.tensor(np.array(train_Y1)).reshape(-1, 1).float()
                val_X1 = torch.tensor(np.array(val_X1)).reshape(-1, val_X1.shape[1]).float()
                val_Y1 = torch.tensor(np.array(val_Y1)).reshape(-1, 1).float()
                test_X1 = torch.tensor(np.array(test_X1)).reshape(-1, test_X1.shape[1]).float()
                test_Y1 = torch.tensor(np.array(test_Y1)).reshape(-1, 1).float()
                
                # training
                print("Start Training")
                pre_time = time.time()
                best_model = training(
                    model = model,
                    train_X = train_X1,
                    train_Y = train_Y1,
                    val_X = val_X1,
                    val_Y = val_Y1,
                    max_epochs = args.max_epochs,
                    patience = args.patience,
                    lr = args.lr,
                    batch_size = args.batch_size,
                    y_scaler = scaler_y,
                    device = device,
                    LAMBDA = 1,
                    verbose = False
                )
                print("Time(s): ",int(time.time() - pre_time))
            
                # evaluation
                best_model.eval()
                with torch.no_grad():
                    w_hat, w_tilde, cluster_indices, y_tilde = best_model(test_X1.to(device))
                if scaler_y:
                    pred = scaler_y.inverse_transform(y_tilde.detach().cpu().numpy())
                    output_loss_test = np.mean((pred - scaler_y.inverse_transform(np.array(test_Y1)))**2)
                else:
                    output_loss_test = torch.mean((y_tilde - test_Y1)**2)
                
                # save inner test performance
                print(f"[Experiments {outer_epoch+1}<{inner_epoch+1}>] [Final MSE] ", output_loss_test , "[Final RMSE] ", output_loss_test**(1/2))
                result_preformance.append(output_loss_test**(1/2))
        
        # save test performance statistics
        print(f"[{data_name}] {np.mean(np.array(result_preformance))} +- {np.std(np.array(result_preformance))}")
        final_result.append([np.mean(np.array(result_preformance)),np.std(np.array(result_preformance))])
    
    # view test performance statistics
    for i, data_name in enumerate(args.dataset_name_list):
        print(f"[{data_name}]: {final_result[i][0]} +- {final_result[i][1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=10000, help="Max Epochs")
    parser.add_argument("--patience", type=int, default=1000, help="Max Patience for Early Stopping")
    parser.add_argument("--proxy_hidden_shape", nargs='+', type=int, default=[64,64,64], help="Proxy Networks Hidden Shape")
    parser.add_argument("--dropout", type=float, default=0.2, help="Proxy Networks Dropout Rate")
    parser.add_argument("--coverage", type=str, default='large', help="Coverage options in {'small','medium','large','xlarge'}")
    parser.add_argument("--num_K", type=int, default=None, help="number of linear models (If num_K is not None, then coverage will ignored)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--dataset_name_list", nargs='+', type=str,  default=['conduct','housing', 'bike', 'electrical', 'plant', 'wine', 'concrete'], help="Dataset Name List in ['conduct','housing', 'bike', 'electrical', 'plant', 'wine', 'concrete']")
    
    args = parser.parse_args()
    
    main(args)