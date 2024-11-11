# 💊 Pill Recognition Research Repository

![Python](https://img.shields.io/badge/python-v3.11-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.2.1-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit-v1.4.0--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-v1.26-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-v2.1.0-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.7.1-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![OpenCV](https://img.shields.io/badge/opencv-4.5.5-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

Welcome to the Pill Recognition Research Repository! This repository hosts code and resources related to 
three conference and a journal paper focusing on advancing pill recognition techniques. 
Each paper corresponds to a separate branch in this repository, and they are briefly introduced below:

<!-- TOC -->
* [💊 Pill Recognition Research Repository](#-pill-recognition-research-repository)
  * [🚨 Repository Status](#-repository-status)
* [📑 Papers](#-papers)
  * [1. Multi-Stream Pill Recognition with Attention](#1-multi-stream-pill-recognition-with-attention)
    * [Abstract:](#abstract)
  * [2. Pill Metrics Learning with Multihead Attention](#2-pill-metrics-learning-with-multihead-attention)
    * [Abstract:](#abstract-1)
  * [3. Word and Image Embeddings in Pill Recognition](#3-word-and-image-embeddings-in-pill-recognition)
    * [Abstract:](#abstract-2)
  * [4. Metric-based pill recognition with the help of textual and visual cues](#4-metric-based-pill-recognition-with-the-help-of-textual-and-visual-cues)
    * [Abstract:](#abstract-3)
* [📈 Evaluation & Usage](#-evaluation--usage)
<!-- TOC -->

## 🚨 Repository Status
As of 2024.11.11.:

| Branch Name  | Status         | Description |
|--------------|----------------|---|
| `idaacs2023` | Up-to-date     |  Multi-Stream Pill Recognition with Attention  |   
| `kdir2023`   | Under revision |  Pill Metrics Learning with Multihead Attention |  
| `visapp2024` | Under revision |  Word and Image Embeddings in Pill Recognition |  
| `wiley`      | Under revision |  Metric-based pill recognition with the help of textual and visual cues | 


# 📑 Papers
## 1. Multi-Stream Pill Recognition with Attention

### Abstract:
We tackle the pill recognition challenge through a groundbreaking approach that employs a multi-stream network 
with EfficientNet-B0 and a self-attention mechanism. To eliminate the explicit training of printed or embossed 
patterns, Local Binary Pattern (LBP) features are utilized. Evaluation is performed on two datasets, demonstrating
that our proposed model surpasses previous models in Top-1 and Top-5 accuracy. Notably, the model also outperforms
the YOLOv7 network in a reference-quality use-case.

**Highlights:**
* Multi-stream network with EfficientNet-B0
* Self-attention for improved feature capture
* Outperforms YOLOv7 in specific use cases

Branch: `idaacs2023`

## 2. Pill Metrics Learning with Multihead Attention

### Abstract:
In the realm of object recognition, especially where new classes can emerge dynamically, few-shot learning holds
significant importance. Our article focuses on metrics learning, a fundamental technique for few-shot object
recognition, successfully applied to pill recognition. We employ multi-stream metrics learning networks and
explore the integration of multihead attention layers at various points in the network. The model's performance
is evaluated on two datasets, showcasing superior results compared to a state-of-the-art multi-stream pill
recognition network.

**Highlights:**
* Few-shot learning with metric learning
* Multihead attention at various network stages
* Superior accuracy compared to previous multi-stream approaches

Branch: `kdir2023`

## 3. Word and Image Embeddings in Pill Recognition

### Abstract:
Addressing the crucial task of improving pill recognition accuracy within a metrics learning framework, our study
introduces a multi-stream visual feature extraction and processing architecture. Leveraging multi-head attention
layers, we estimate pill similarity. An innovative enhancement to the triplet loss function incorporates word 
embeddings, injecting textual pill similarity into the visual model. This refinement operates on a finer scale 
than conventional triplet loss models, resulting in enhanced visual model accuracy. Experiments and evaluations
are conducted on a new, freely available pill dataset.

**Highlights:**
* Multi-stream architecture with visual and text embeddings
* Enhanced triplet loss for better visual model performance
* Freely accessible pill dataset for further experimentation

Branch: `visapp2024`

## 4. Metric-based pill recognition with the help of textual and visual cues

### Abstract:
Pill image recognition by machine vision can reduce the risk of taking the wrong medications, a severe healthcare
problem. Automated dispensing machines or home applications both need reliable image processing techniques
to compete with the problem of changing viewing conditions, large number of classes, and the similarity in pill
appearance. We attack the problem with a multi-stream, two-phase metric embedding neural model. To enhance
the metric learning procedure, we introduce dynamic margin setting into the loss function. Moreover, we show
that besides the visual features of drug samples, even free text of drug leaflets (processed with a natural language
model) can be used to set the value of the margin in the triplet loss and thus increase the recognition accuracy of
testing. Thus, besides using the conventional metric learning approach, the given discriminating features can be
explicitly injected into the metric model using the NLP of the free text of pill leaflets or descriptors of images of
selected pills. We analyse the performance on two datasets and report a 1.6% (two-sided) and 2.89% (one-sided)
increase in Top-1 accuracy on the CURE dataset compared to existing best results. The inference time on CPU and
GPU makes the proposed model suitable for different kinds of applications in medical pill verification; moreover,
the approach applies to other areas of object recognition where few-shot problems arise. The proposed high-level
feature injection method (into a low-level metric learning model) can also be exploited in other cases, where class
features can be well described with textual or visual cues.

**Highlights:**
* Multi-stream neural model with dynamic margin setting
* Combines visual cues with NLP-processed textual data

Branch: `journal`

# 📈 Evaluation & Usage
Each branch contains detailed instructions for reproducing experiments, including pre-processing steps, model training,
and evaluation scripts. Please refer to individual branch documentation for usage examples and dataset preparation.
