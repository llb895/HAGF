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
from data_pipeline import HighDimDataset_PRJNA929650
from data_standardization import extract_features_from_dsets,get_scalers,feature_standardization,update_datasets_with_standardized_features

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold



def main():
    set_seed(999)
    
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_paths_train = [
        "../data/dataset/PRJNA929650_cross/end_motifs.csv",
        "../data/dataset/PRJNA929650_cross/FSR.csv",
        "../data/dataset/PRJNA929650_cross/methy_new_standard.csv",
        "../data/dataset/PRJNA929650_cross/CopyNumber.csv"  
    ]

    csv_paths_val = [
        "../data/dataset/PRJNA929650_validation/end_motifs.csv",
        "../data/dataset/PRJNA929650_validation/FSR.csv",
        "../data/dataset/PRJNA929650_validation/methy_new_standard.csv",
        "../data/dataset/PRJNA929650_validation/CopyNumber.csv"  
    ]
    datasets_test = []
    datasets_train = []
    dfs = {}
    train_dataset_list=[]
    val_dataset_list=[]
    test_dataset_list=[]

    
    for path in csv_paths_train:
        key = path.split('/')[-1].rsplit('.', 1)[0]
        dfs[key] = pd.read_csv(path).reset_index(drop=True)
        datasets_train.append(HighDimDataset_PRJNA929650(dfs[key]))


    for path in csv_paths_val:
        key = path.split('/')[-1].rsplit('.', 1)[0]
        dfs[key] = pd.read_csv(path).reset_index(drop=True)
        datasets_test.append(HighDimDataset_PRJNA929650(dfs[key]))



    full_length = len(datasets_train[0])  
    val_length = int(full_length * 0.2)
    train_length = full_length - val_length
    
    train_subset, val_subset = random_split(range(full_length), [train_length, val_length])
    train_idx = list(train_subset)
    val_idx = list(val_subset)
    
    
    
    test_idx = list(range(len(datasets_test[0])))


    # Hyper-parameters
    batch_size = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters (example)
    input_sizes = [df.shape[1] - 3 for df in dfs.values()]
    model_params = {
        'input_sizes': input_sizes,
        'num_layers': 2,
        'hidden_size': 100,
        'output_size': 2,
        'dropout': 0,
        'num_masks': 2
    }
    
    test_aucs = []
    all_fold_aucs = []
    
    

    # Create Subsets for each modality
    train_dsets = [Subset(ds, train_idx) for ds in datasets_train]
    val_dsets   = [Subset(ds, val_idx)   for ds in datasets_train]
    test_dsets  = [Subset(ds, test_idx)  for ds in datasets_test]


    #--------------------------------------------------------------------------------------------------------------       
    train_features = extract_features_from_dsets(train_dsets)
    val_features = extract_features_from_dsets(val_dsets)
    test_features = extract_features_from_dsets(test_dsets)
    
    scalers = get_scalers(train_features,val_features)
    train_features,val_features,test_features = feature_standardization(train_features,val_features,test_features,scalers)


    for i in range(len(datasets_train)):
        data = datasets_train[i].data 
        new_data = data.clone()
        new_data[train_idx] = train_features[i]
        new_data[val_idx]   = val_features[i]
        datasets_train[i].data = new_data

    train_dsets = [Subset(ds, train_idx) for ds in datasets_train]
    val_dsets   = [Subset(ds, val_idx)   for ds in datasets_train]
    
    
    for i in range(len(datasets_test)):
        data = datasets_test[i].data  
        new_data = data.clone()
        new_data[test_idx] = test_features[i]
        
        datasets_test[i].data = new_data    
    test_dsets  = [Subset(ds, test_idx)  for ds in datasets_test]



    
    
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
    
    #test_ID = test_ID[0]
    test_ID = [item for sublist in test_ID for item in sublist]

    df = pd.DataFrame({
        'ID': test_ID,
        'Label': test_labels,
        'Probability': test_probs
    })
    
    save_data(df)

    all_fold_aucs.append(test_auc)
    
    # After CV, summarize results
    mean_auc = sum(all_fold_aucs) / len(all_fold_aucs)
    print("AUC:", mean_auc)


def set_seed(seed=999):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_data(df):
    df.to_csv(f"../result/PRJNA929650/val/val.csv", index=False)
    
if __name__ == '__main__':
    main()
        