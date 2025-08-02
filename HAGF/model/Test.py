# coding=gb2312
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F


def Test(model,test_loaders):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    
    test_labels_all = []
    test_probs_all  = []
    test_ID_all =[]
    
    test_steps = min(len(loader) for loader in test_loaders)
    
    with torch.no_grad():
        for batches in zip(*test_loaders):
            
            ID = batches[0][2]
            test_ID_all.append(ID)
            
            
            datas  = [b[0].to(device) for b in batches]  # list of [B, D_i]
            labels = batches[0][1].to(device)            # [B]
    
            logits = model(datas)                        # [B, num_classes]
            loss   = criterion(logits, labels)
            test_loss += loss.item()
    
            #Convert with Softmax to probabilities, then take the probability of the positive class (index = 1).
            probs = F.softmax(logits, dim=1)             # [B, 2]
            test_labels_all.append(labels.cpu())
            test_probs_all.append(probs[:, 1].cpu())
    
    test_labels_all = torch.cat(test_labels_all).numpy()
    test_probs_all  = torch.cat(test_probs_all).numpy()
    
    print(test_labels_all)
    print(test_probs_all)
    
    # average valdataset AUC
    avg_test_loss = test_loss / test_steps
    test_auc      = roc_auc_score(test_labels_all, test_probs_all)
    
    print(f"Test Loss: {avg_test_loss:.4f}, Test AUC: {test_auc:.4f}")
    return test_labels_all,test_probs_all,test_auc, test_ID_all
    
    
    
    
    