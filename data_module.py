import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo
from sklearn import datasets
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

def load_data(csv_name: str):
    """
    input
        - csv_name: str {'conduct', 'housing', 'bike', 'electrical', 'plant', 'wine', 'concrete'}
    output
        - X: torch.tensor (float32, (N,p)) [covariate vectors]
        - Y: torch.tensor (float32, (N,1)) [responses]
    """
    
    if csv_name == 'conduct':
        superconduct = fetch_ucirepo(id=464) 
        X = superconduct.data.features 
        Y = superconduct.data.targets 
    
    elif csv_name == 'housing':
        ca_housing = datasets.fetch_california_housing()
        X = pd.DataFrame(ca_housing.data, columns=ca_housing.feature_names)
        Y = pd.DataFrame(ca_housing.target, columns=ca_housing.target_names)

    elif csv_name == 'bike':
        bike_sharing = fetch_ucirepo(id=275) 
        X = bike_sharing.data.features.drop(['dteday'],axis=1)  
        ###########################################
        """
        Categorical Feature Preprocessing
        """
        encoder = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, min_frequency=1, drop='first')
        catcolumns = ['season','yr','mnth','hr','holiday','weekday','workingday']
        numcolumns = sorted(list(set(X.columns)-set(catcolumns)))
        cat_X = encoder.fit_transform(X[catcolumns])
        num_X = np.array(X[numcolumns])
        ###########################################
        X = np.concatenate([cat_X, num_X], axis=1)
        Y = bike_sharing.data.targets
    
    elif csv_name == 'electrical':
        electrical_grid_stability_simulated_data = fetch_ucirepo(id=471) 
        X = electrical_grid_stability_simulated_data.data.features 
        Y = electrical_grid_stability_simulated_data.data.targets.drop(['stabf'], axis=1)        

    elif csv_name == 'plant':
        plant = fetch_ucirepo(id=294) 
        X = plant.data.features 
        Y = plant.data.targets

    elif csv_name == 'wine':
        wine_quality = fetch_ucirepo(id=186) 
        X = wine_quality.data.features 
        Y = wine_quality.data.targets
    
    elif csv_name == 'concrete':
        concrete = fetch_ucirepo(id=165) 
        X = concrete.data.features 
        Y = concrete.data.targets

    X = torch.tensor(np.array(X)).reshape(-1, X.shape[1]).float()
    Y = torch.tensor(np.array(Y)).reshape(-1, 1).float()
    return X, Y
