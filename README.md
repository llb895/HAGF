# Hierarchically Adaptive Gated Fusion for Multimodal cfDNA Feature Integration Enables High-Accuracy Cancer Detection
## Introduction
Cell-free DNA (cfDNA) methylation, copy number variations (CNV), and fragmentation patterns are distinct genomic and epigenetic biomarkers with strong potential for early cancer detection. Integrating these modalities can significantly enhance diagnostic accuracy, but their high heterogeneity challenges effective multimodal fusion. To address this, we propose the Hierarchical Adaptive Gating Fusion network (HAGF), which excels in high-dimensional learning with limited samples. Its adaptive gating mechanism dynamically assigns importance weights to each feature dimension, ensuring robust biological interpretability. Additionally, HAGF enables deep, abstract representation learning across diverse feature types, fully capturing each modality’s intrinsic patterns. Validated on large-scale, diverse cancer datasets, HAGF demonstrates superior discriminative power and generalization. These findings confirm HAGF’s ability to integrate multimodal cfDNA features for high-accuracy tumor detection, providing a robust foundation for a versatile, multi-cancer liquid biopsy assay.


## Overview
<div align=center>
<img src="https://github.com/llb895/HAGF/blob/main/HAGF/fig1.jpg">
</div>

## Table of Contents
* | [1 Environment](#section1) |<br>
* | [2 Preparation](#section2) |<br>
* | [3 Model Prediction](#section3) |<br>
* | [4 Output Results](#section4) |<br>
* | [5 Citation](#section5) |<br>
* | [6 References](#section6) |<br>


<a id="section1"></a>
## 1 Environment
We used Python 3.7 for our experiments, and our CUDA version was 11.8. 
To set up the environment:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

<a id="section2"></a>
## 2 Preparation
In this study, we demonstrate the functionality of HAGF through a case study. Using datasets from Zhang et al. dataset[^1] and Pham et al. dataset[^2], we employed features from three different modalities for both independent and cross-validation.
```
project
│   README.md
│   
└───HAGF
   │   readme.md
   │
   └───model
   │   │
   │   │   ...    
   │
   └───dataset
   │   │
   │   │   ...
   │ 
   └───result
   │   │
   │   │   ...
```
The ***model*** directory stores HAGF model code and related data processing tools. <br>
The ***dataset*** directory contains raw sample data. <br>
The ***result*** directory contains the predicted output matrix.<br>

<a id="section3"></a>
## 3 Model Prediction
### In cross-validation
First, enter the model folder.
```
cd /ELSM/model/
```
Validate the model on Zhang et al.'s dataset.
```
python Cancer_Prediction_CRA001537.py 
```
Perform cross-validation of the model on Pham et al.'s dataset.
```
python Cancer_Prediction_cross_PRJNA929650.py
```
Perform independent-validation of the model on Pham et al.'s dataset.
```
python Cancer_Prediction_val_PRJNA929650.py
```
The results are stored in the **result** folder.

<a id="section4"></a>
## 4 Output Results
The prediction results are stored in the ***result*** folder.


<a id="section5"></a>
## 5 Cite Us
If you use **HAGF** framework in your own studies, and work, please cite it by using the following:
```
@article{HAGF,
    title={Hierarchically Adaptive Gated Fusion for Multimodal cfDNA Feature Integration Enables High-Accuracy Cancer Detection},
    author={Libo Lu, ..., and Xionghui Zhou},
    year={2025},
}
```

<a id="section6"></a>
## 6 References
[^1]:Zhang H, Dong P, Guo S, et al. Hypomethylation in HBV integration regions aids non-invasive surveillance to hepatocellular carcinoma by low-pass genome-wide bisulfite sequencing[J]. BMC medicine, 2020, 18: 1-14.
[^2]:Pham T M Q, Phan T H, Jasmine T X, et al. Multimodal analysis of genome-wide methylation, copy number aberrations, and end motif signatures enhances detection of early-stage breast cancer[J]. Frontiers in Oncology, 2023, 13: 1127086.
