# Enhanced Early Cancer Detection via Multi-Omics cfDNA Fragmentation Integration Using an Early-Late Fusion Neural Network with Sample-Modality Evaluation
## Introduction
Multi-omics cfDNA fragmentation patterns show promise as biomarkers for early cancer detection, but fusing their multimodal data faces challenges from heterogeneity and small sample sizes. We propose ELSM, a framework integrating two-stage neural network fusion with sample-modality evaluation to effectively combine 13 cfDNA fragmentomic features. Evaluated across five datasets (1,994 samples, 10 cancer types), ELSM outperforms both unimodal classifiers and state-of-the-art fusion models in cancer detection and tissue-of-origin prediction. Biological analysis of key genomic regions linked to high-contributing modalities validates biological relevance, highlighting ELSM’s clinical utility.

## Overview
<div align=center>
<img src="https://github.com/llb895/ELSM/blob/main/Fig/1.png">
</div>

## Table of Contents
| [1 Environment](#section1) |<br>
| [2 Preparation](#section2) |<br>
| [3 Modality Evaluation](#section3) |<br>
| [4 Model Prediction](#section4) |<br>
| [5 Output Results](#section5) |<br>
| [6 Citation](#section6) |<br>
| [7 References](#section7) |<br>


<a id="section1"></a>
## 1 Environment
We used Python 3.7 for our experiments, and our CUDA version was 11.8. 
To set up the environment:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

<a id="section2"></a>
## 2 Preparation
In this study, we demonstrate the functionality of ELSM through a case study that performs independent validation and cross-validation based on 13 fragmentomic features from Mathios et al.’s independent dataset and the Mathios et al.’s LUCAS dataset.[^1]
```
project
│   README.md
│   
└───ELSM
   │   readme.txt
   │
   └───sample_level_evaluation_strategy_result
   │   │
   │   │   ...    
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
The ***dataset*** directory contains raw sample data. <br>
The ***model*** directory stores ELSM model code and related data processing tools. <br>
The ***sample_level_evaluation_strategy_result*** directory holds sampling-resampled data.<br>
The ***result*** directory contains the predicted output matrix.<br>

<a id="section3"></a>
## 3 Modality Evaluation
### In cross-validation
### Enter the model folder.
```
cd /ELSM/model/
```
### Execute the ***sample_level_evaluation_strategy_cross.py*** file.
```
python sample_level_evaluation_strategy_cross.py "../dataset/10-fold-cross-validation/" "../sample_level_evaluation_strategy_result/"
```
The processed results will then be available in ***ELSM/sample_level_evaluation_strategy_result/***.<br>
Similarly, this is applicable to independent validation.

<a id="section4"></a>
## 4 Model Prediction
### In cross-validation
For the sample-level resampled data, model predictions are performed using an early-late fusion neural network.
```
cd /ELSM/model/
python execution_cross.py "../sample_level_evaluation_strategy_result/" 
```
<br>
Similarly, this is applicable to independent validation.

<a id="section5"></a>
## 5 Output Results
The prediction results are stored in the ***result*** folder.


<a id="section6"></a>
## 6 Cite Us
If you use **ELSM** framework in your own studies, and work, please cite it by using the following:
```
@article{ELSM,
    title={Enhanced Early Cancer Detection via Multi-Omics cfDNA Fragmentation Integration Using an Early-Late Fusion Neural Network with Sample-Modality Evaluation},
    author={Libo Lu, ..., and Xionghui Zhou},
    year={2025},
}
```

<a id="section7"></a>
## 7 References
[^1]:D. Mathios, J.S. Johansen, S. Cristiano, J.E. Medina, J. Phallen, K.R. Larsen, D.C. Bruhm, N. Niknafs, L. Ferreira, V.J.N.c. Adleff, Detection and characterization of lung cancer using cell-free DNA fragmentomes, 12 (2021) 5060.
