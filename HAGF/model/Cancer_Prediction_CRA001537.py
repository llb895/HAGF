# coding=gb2312
import sys
sys.path.append("../model/")

from model import CrossModalFusionClassifier
from Train import Train
from Test import Test
import os
import torch
import numpy as np
from torch.utils.data import random_split,Dataset, DataLoader, Subset
import random
from sklearn.model_selection import KFold, train_test_split
import pandas as pd

sys.path.append("../data/")
from data_pipeline import HighDimDataset_CRA001537
from data_standardization import extract_features_from_dsets,get_scalers,feature_standardization,update_datasets_with_standardized_features

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold



def main():
    set_seed(999)
    
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_paths = [
        "../data/dataset/CRA001537_cross/end_motifs.csv",
        "../data/dataset/CRA001537_cross/FSR.csv",
        "../data/dataset/CRA001537_cross/methy_selected.csv",
        "../data/dataset/CRA001537_cross/CopyNumber.csv"  
    ]
    
    datasets = []
    dfs = {}
    train_dataset_list=[]
    val_dataset_list=[]
    test_dataset_list=[]
    
    for path in csv_paths:
        key = path.split('/')[-1].rsplit('.', 1)[0]
        dfs[key] = pd.read_csv(path)
        datasets.append(HighDimDataset_CRA001537(dfs[key]))
    
    # Hyper-parameters
    n_repeats = 5
    n_splits = 10
    val_ratio = 0.2  # ratio for validation split within each fold's training data
    batch_size = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters (example)
    input_sizes = [df.shape[1] - 3 for df in dfs.values()]
    
    '''
    model_params = {
        'input_sizes': input_sizes,
        'num_layers': 2,
        'hidden_size': 100,
        'output_size': 2,
        'dropout': 0,
        'num_masks': 2,
        'dropout': 0
    }
    
    
    
    # Storage for metrics
    test_aucs = []
    all_fold_aucs = []
    
    # Outer loop: repeat cross-validation
    for repeat in range(n_repeats):
        print(f"Repeat {repeat+1}/{n_repeats}")
        #kf = KFold(n_splits=n_splits, shuffle=True, random_state=999 + repeat)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=999 + repeat)
        
        #for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(datasets[0]), start=1):
        labels_all = datasets[0].labels.numpy()  # 提前取出标签
        for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(datasets[0], labels_all), start=1):            
            
            print(f" Fold {fold_idx}/{n_splits}")
            
            # Further split train_val_idx into train and val subsets
            labels_all = datasets[0].labels.numpy()  
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio,
                shuffle=True,
                stratify=labels_all[train_val_idx],  # 注意这里传的是子集的标签
                random_state=999 + repeat + fold_idx
            )
            
    
            # Create Subsets for each modality
            train_dsets = [Subset(ds, train_idx) for ds in datasets]
            val_dsets   = [Subset(ds, val_idx)   for ds in datasets]
            test_dsets  = [Subset(ds, test_idx)  for ds in datasets]
    
    
    
            #--------------------------------------------------------------------------------------------------------------       
            train_features = extract_features_from_dsets(train_dsets)
            val_features = extract_features_from_dsets(val_dsets)
            test_features = extract_features_from_dsets(test_dsets)
            
            scalers = get_scalers(train_features,val_features)
            train_features,val_features,test_features = feature_standardization(train_features,val_features,test_features,scalers)

            train_dsets, val_dsets, test_dsets = update_datasets_with_standardized_features(
                datasets, train_idx, val_idx, test_idx,
                train_features, val_features, test_features
            )
            #--------------------------------------------------------------------------------------------------------------    
          
    
            # DataLoaders
            train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True)  for ds in train_dsets]
            val_loaders   = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in val_dsets]
            test_loaders  = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in test_dsets]
    
            
            # Initialize model for this fold
            model = CrossModalFusionClassifier(**model_params).to(device)
    
            # Train on train_loaders and validate on val_loaders
            model = Train(model, train_loaders, val_loaders)
    
            # Test and collect AUC
            test_labels, test_probs, test_auc,test_ID = Test(model, test_loaders)
            
            test_ID = test_ID[0]
            df = pd.DataFrame({
                'ID': test_ID,
                'Label': test_labels,
                'Probability': test_probs
            })
            
            #save_data(df,repeat,fold_idx)

            all_fold_aucs.append(test_auc)
    
    # After CV, summarize results
    mean_auc = sum(all_fold_aucs) / len(all_fold_aucs)
    print("5×10 CV mean AUC:", mean_auc)
    '''


    from itertools import product
    
    # 定义超参数搜索空间
    hidden_sizes = [50, 100, 150, 200]
    num_layers_list = [1, 2, 3]
    dropouts = [0.0, 0.1 , 0.2]
    num_masks_list = [1, 2 , 3]
    
    # 记录最佳参数和对应AUC
    best_auc = -1
    best_params = None
    
    
    
    # 遍历所有组合
    for hidden_size, num_layers, dropout, num_masks in product(hidden_sizes, num_layers_list, dropouts, num_masks_list):
        print(f"Testing: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}, num_masks={num_masks}")
        
        
        model_params = {
            'input_sizes': input_sizes,
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'output_size': 2,
            'dropout': dropout,
            'num_masks': num_masks
        }
    
        all_fold_aucs = []
        for repeat in range(n_repeats):
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=999 + repeat)
            labels_all = datasets[0].labels.numpy()
    
            for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(datasets[0], labels_all), start=1):
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_ratio,
                    shuffle=True,
                    stratify=labels_all[train_val_idx],
                    random_state=999 + repeat + fold_idx
                )
    
                train_dsets = [Subset(ds, train_idx) for ds in datasets]
                val_dsets   = [Subset(ds, val_idx)   for ds in datasets]
                test_dsets  = [Subset(ds, test_idx)  for ds in datasets]
    
                train_features = extract_features_from_dsets(train_dsets)
                val_features   = extract_features_from_dsets(val_dsets)
                test_features  = extract_features_from_dsets(test_dsets)
    
                scalers = get_scalers(train_features, val_features)
                train_features, val_features, test_features = feature_standardization(
                    train_features, val_features, test_features, scalers
                )
    
                train_dsets, val_dsets, test_dsets = update_datasets_with_standardized_features(
                    datasets, train_idx, val_idx, test_idx,
                    train_features, val_features, test_features
                )
    
                train_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in train_dsets]
                val_loaders   = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in val_dsets]
                test_loaders  = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in test_dsets]
    
                model = CrossModalFusionClassifier(**model_params).to(device)
                model = Train(model, train_loaders, val_loaders)
    
                test_labels, test_probs, test_auc, _ = Test(model, test_loaders)
                all_fold_aucs.append(test_auc)
    
        mean_auc = sum(all_fold_aucs) / len(all_fold_aucs)
        print(f"Mean AUC for current setting: {mean_auc:.4f}")
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = model_params.copy()
    
    print("\nBest AUC:", best_auc)
    print("Best params:", best_params)



    # 定义保存路径
    save_dir = "/home/lulb/SUBJECT/Methylation/SUBJECT/CRA001537/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model_params.txt")
    
    # 写入内容
    with open(save_path, 'w') as f:
        f.write(f"Best AUC: {best_auc:.6f}\n")
        f.write("Best Parameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nBest parameters saved to {save_path}")



def set_seed(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_data(df,repeat,fold_idx):
    df.to_csv(f"../result/CRA001537/{repeat+1}_{fold_idx}_fold.csv", index=False)
 
    
if __name__ == '__main__':
    main()
 
 
 
        