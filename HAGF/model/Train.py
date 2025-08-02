# coding=gb2312
from sklearn.metrics import roc_auc_score
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
import copy

config = {
    "num_classes": 2,
    "batch_size": 10,
    "num_epochs": 50,
    "lr": 0.001,     
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def Train(model,train_loaders,val_loaders): 
   
    best_model = model
    best_val_loss = 999999
    counter = 0
    patience = 5   # 5 = 0.8175  
    delta = 1e-5

    train_steps = min(len(loader) for loader in train_loaders)
    val_steps   = min(len(loader) for loader in val_loaders)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss() 

    for epoch in range(config["num_epochs"]):
        
        # —— train —— #
        model.train()
        total_loss = 0.0
    
        train_labels_all = []
        train_probs_all  = []
    
        for batches in zip(*train_loaders):
            # prepare dataset
            
            datas  = [b[0].to(device) for b in batches]  
            
            
            labels = batches[0][1].to(device)            

            optimizer.zero_grad()
            logits = model(datas)                       
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
            probs = F.softmax(logits, dim=1)          
    
            train_labels_all.append(labels.detach().cpu())

            train_probs_all.append(probs[:, 1].detach().cpu())
    
        train_labels_all = torch.cat(train_labels_all).numpy()
        train_probs_all  = torch.cat(train_probs_all).numpy()
    
        # average train loss
        avg_train_loss = total_loss / train_steps



        # train dataset AUC
        train_auc = roc_auc_score(train_labels_all, train_probs_all)
        
    
        # —— val —— #
        model.eval()
        val_loss = 0.0
        val_labels_all = []
        val_probs_all  = []
    
        with torch.no_grad():
            for batches in zip(*val_loaders):
                datas  = [b[0].to(device) for b in batches]
                labels = batches[0][1].to(device)
    
                logits = model(datas)
                loss   = criterion(logits, labels)
                val_loss += loss.item()
    
                probs = F.softmax(logits, dim=1)
                val_labels_all.append(labels.cpu())
                val_probs_all.append(probs[:, 1].cpu())
    
        val_labels_all = torch.cat(val_labels_all).numpy()
        val_probs_all  = torch.cat(val_probs_all).numpy()
    
        print("------------------------------")
        avg_val_loss = val_loss / val_steps
        val_auc = roc_auc_score(val_labels_all, val_probs_all)
   
        print(
            f"Epoch {epoch+1}/{config['num_epochs']} | "
            f"Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f} | "
            f"Val   Loss: {avg_val_loss:.4f}, Val   AUC: {val_auc:.4f}"
        )

        best_val_loss, counter, stop = check_early_stop(
            current_val_loss=avg_val_loss,
            best_val_loss=best_val_loss,
            counter=counter,
            patience=patience,
            delta=delta
        )
        if avg_val_loss < best_val_loss:
            best_model = model
            
        if stop:
            print(f"Epoch {epoch+1}: no loss improvement for {patience} epochs. Early stopping.")
            break
    
    return best_model  
    
def check_early_stop(current_val_loss, best_val_loss, counter, patience, delta=0.0):

    if current_val_loss < best_val_loss - delta:
        best_val_loss = current_val_loss
        counter = 0
        early_stop = False
    else:
        counter += 1
        early_stop = (counter >= patience)
    return best_val_loss, counter, early_stop
    
    
    
    
        
    
    
      