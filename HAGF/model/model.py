# coding=gb2312
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#The entmax function computes a generalized softmax that produces sparse probability distributions controlled by the parameter alpha.
def entmax(alpha, x):
    x = x - torch.max(x, dim=-1, keepdim=True)[0]
    e_x = torch.exp(x)
    s = (torch.sum(e_x ** alpha, dim=-1, keepdim=True) + 1e-5) ** (1 / alpha)    
    return e_x / s


#divides the input features into groups, applies multiple learnable sparse masks per group to select features, then processes each masked group through attention-like transformations before aggregating and concatenating the results.
class DynamicFeatureGroupingLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_masks,group_ratio=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.group_ratio = group_ratio
        self.num_masks = num_masks  

        self.base_group_size = max(1, int(input_size * group_ratio))
        self.num_groups = (input_size + self.base_group_size - 1) // self.base_group_size
       
        self.W_masks = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.randn(
                    self.base_group_size if i < self.num_groups-1 
                    else input_size - i*self.base_group_size
                )) 
                for _ in range(num_masks)
            ]) for i in range(self.num_groups)
        ])
        
        self.W1s = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, 
                self.base_group_size if i < self.num_groups-1 
                else input_size - i*self.base_group_size
            )) 
            for i in range(self.num_groups)
        ])
        self.W2s = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size, 
                self.base_group_size if i < self.num_groups-1 
                else input_size - i*self.base_group_size
            )) 
            for i in range(self.num_groups)
        ])
        
        self.bn1s = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.num_groups)])
        self.bn2s = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.num_groups)])


    def forward(self, x):
        
        self.all_masks_concat = [[] for _ in range(self.num_masks)]  
        
        outputs = []
        for i in range(self.num_groups):
            start = i * self.base_group_size
            end = start + (self.base_group_size if i < self.num_groups-1 
                          else self.input_size - i*self.base_group_size)
            x_part = x[:, start:end]
            
            group_output = 0
            for k in range(self.num_masks):
                mask = entmax(1.1, self.W_masks[i][k])

                self.all_masks_concat[k].append(mask.detach().cpu())                
                
                weighted = x_part * mask
                h1 = F.linear(weighted, self.W1s[i])
                h2 = F.linear(weighted, self.W2s[i])
                gate = torch.sigmoid(self.bn1s[i](h1))
                activated = F.relu(gate * self.bn2s[i](h2))
                group_output += activated 

            outputs.append(group_output)
        
        self.all_masks_concat = torch.stack(
            [torch.cat(mask_parts, dim=0) for mask_parts in self.all_masks_concat], dim=0
        )   
        return torch.cat(outputs, dim=1)
                

#progressively abstracts input features through stacked dynamic grouping layers, then combines the final transformed features with a nonlinear projection of the original input for enhanced representation.        
class HierarchicalFeatureExtractor(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size, dropout,num_masks,
                 group_ratio):
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_size = output_size
        current_size = input_size
        self.orig_fc = nn.LazyLinear(self.output_size)
        for _ in range(num_layers):
            layer = DynamicFeatureGroupingLayer(
                input_size=current_size,
                hidden_size=hidden_size,
                group_ratio=group_ratio,
                num_masks=num_masks
            )
            self.layers.append(layer)
            current_size = hidden_size * layer.num_groups  

        self.final_fc = nn.Linear(current_size, output_size)
        self.dropout = nn.Dropout(0)  

    def forward(self, x):
        self.input_dim = x.shape[1]
        original_feat = x
        
        for layer in self.layers:
            x = self.dropout(F.gelu(layer(x))) 

        out = self.final_fc(x)
        orig_proj = self.orig_fc(original_feat) 
        orig_proj = F.gelu(orig_proj)            
        out = torch.cat([out, orig_proj], dim=1)  
        return out


#extracts features from multiple input modalities using separate hierarchical extractors, concatenates their outputs, and performs final classification through a fusion neural network.
class CrossModalFusionClassifier(nn.Module):        
    def __init__(
        self,
        input_sizes: list,
        num_layers: int,
        hidden_size: int,
        output_size: int,
        num_masks: int,
        group_ratio=0.1,
        dropout=int
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            HierarchicalFeatureExtractor(
                input_size=sz,
                num_layers=num_layers,
                hidden_size=hidden_size,
                group_ratio=group_ratio,
                num_masks=num_masks,
                dropout=dropout,
                output_size = output_size
            ) for sz in input_sizes
        ])
     
        fusion_dim = sum(branch.output_size for branch in self.branches)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim*2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, inputs: list):
        feats = []
        for idx, (branch, x) in enumerate(zip(self.branches, inputs)):
            feat = branch(x)  
            feats.append(feat)
        fusion = torch.cat(feats, dim=1)
        out = self.classifier(fusion)
        return out



 
 
 
 
 
 