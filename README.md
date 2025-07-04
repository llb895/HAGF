# 基于多模态cfDNA特征的通用型液体活检框架实现高灵敏HCC与乳腺癌诊断
## Introduction
cfDNA甲基化修饰、拷贝数变异（CNA）和cfDNA片段模式分别表征了表观遗传信息的不同维度，均为极具前景的早期癌症检测生物标志物。整合三者有望显著提升癌症诊断性能；然而，其固有的高度异质性给多模态数据融合带来了巨大挑战。为了克服这些限制，有效整合cfDNA中蕴含的互补多模态信息，本研究提出了层次化自适应门控融合网络 (HAGF)。该模型的核心创新在于其强大的高维小样本学习能力；其自适应门控机制可动态学习并输出每一特征维度的重要性权重，为模型决策提供良好的生物学解释性；HAGF能够对不同类型的特征进行深层次的高维抽象表征学习，充分挖掘每种模态数据的内部模式。在HCC和大型乳腺癌队列中，HAGF展现出优异的区分能力和普适性。这些结果充分证明HAGF框架能够有效整合多模态cfDNA特征，实现高精度肿瘤检测，为开发适用于多种癌症的通用型液体活检工具奠定了坚实技术基础。


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
    title={基于多模态cfDNA特征的通用型液体活检框架实现高灵敏HCC与乳腺癌诊断},
    author={Libo Lu, ..., and Xionghui Zhou},
    year={2025},
}
```

<a id="section6"></a>
## 6 References
[^1]:Zhang H, Dong P, Guo S, et al. Hypomethylation in HBV integration regions aids non-invasive surveillance to hepatocellular carcinoma by low-pass genome-wide bisulfite sequencing[J]. BMC medicine, 2020, 18: 1-14.
[^2]:Pham T M Q, Phan T H, Jasmine T X, et al. Multimodal analysis of genome-wide methylation, copy number aberrations, and end motif signatures enhances detection of early-stage breast cancer[J]. Frontiers in Oncology, 2023, 13: 1127086.
