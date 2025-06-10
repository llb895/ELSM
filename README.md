# Enhanced Early Cancer Detection via Multi-Omics cfDNA Fragmentation Integration Using an Early-Late Fusion Neural Network with Sample-Modality Evaluation
## Introduction
Multi-omics cfDNA fragmentation patterns show promise as biomarkers for early cancer detection, but fusing their multimodal data faces challenges from heterogeneity and small sample sizes. We propose ELSM, a framework integrating two-stage neural network fusion with sample-modality evaluation to effectively combine 13 cfDNA fragmentomic features. Evaluated across five datasets (1,994 samples, 10 cancer types), ELSM outperforms both unimodal classifiers and state-of-the-art fusion models in cancer detection and tissue-of-origin prediction. Biological analysis of key genomic regions linked to high-contributing modalities validates biological relevance, highlighting ELSM’s clinical utility.

## Overview
<div align=center>
<img src="https://github.com/llb895/ELSM/blob/main/Fig/1.png">
</div>

## Table of Contents
| [1 Environment](#section1) |
| [2 Preparation](#section2) |
| [3 Modality Evaluation](#section3) |
| [4 Model Prediction](#section4) |
| [5 Output Results](#section5) |


<a id="section1"></a>
## 1 Environment
We used Python 3.7 for our experiments, and our CUDA version was 11.8. 
To set up the environment:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
