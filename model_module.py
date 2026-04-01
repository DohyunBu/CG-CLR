import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class CGCLR(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expert_num: int,
        output_dim: int,
        proxy_hidden_shape: list,
        dropout: float,
        device = 'cpu',

    ) -> None:
        super(CGCLR, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expert_num = expert_num
        self.proxy_hidden_shape = proxy_hidden_shape
        self.dropout = dropout
        self.device = device 

        # Codebook Define
        self.codebook = nn.Embedding(self.expert_num, self.input_dim+1).to(self.device)
        
        # Proxy Define
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.proxy_hidden_shape:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.input_dim+1))
        self.proxy = nn.Sequential(*layers).to(self.device)
        
        # Init
        self.codebook.weight.data.uniform_(-1/self.expert_num, 1/self.expert_num)
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    nn.init.trunc_normal_(m.bias, mean=0, std=0.1)
        self.proxy.apply(weights_init)

    def forward(self, input_tensor) -> torch.Tensor:
        # augmented covariates [x, 1]
        augmented_tensor = torch.cat([input_tensor, torch.ones((input_tensor.shape[0],1), device=self.device).type(torch.float32)], axis=1)
        
        # proxy output
        w_hat = self.proxy(input_tensor) 
        
        # cluster assignment z
        w_hat_flattened = w_hat.view(-1, self.input_dim+1) 
        distances = ((augmented_tensor * w_hat_flattened).sum(axis=1).reshape(-1,1) - (augmented_tensor @ self.codebook.weight.t()))**2
        cluster_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # zth w_tilde
        w_tilde = self.codebook(cluster_indices).view(w_hat.shape)
        
        # codebook prediction
        y_tilde = (augmented_tensor * (w_hat + (- w_hat + w_tilde).detach())).sum(axis=1).reshape(-1,1)
        
        return w_hat, w_tilde, cluster_indices, y_tilde