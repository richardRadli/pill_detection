# Pill Recognition Research Repository

## ⚠️ Important Notice

> **Branch Status Update**  
> - `idaacs2023` and `kdir2023`: **Up to date**  
> - Other branches: **Undergoing revision**  
>   
> Please use the specified branches for stable, current functionality. Changes in other branches may impact code stability until rewrites are complete.

Welcome to the Pill Recognition Research Repository! This repository hosts code and resources related to 
three articles focusing on advancing pill recognition techniques. Each article corresponds to a separate branch 
in this repository, and they are briefly introduced below:

## 1. Multi-Stream Pill Recognition with Attention

### Abstract:
We tackle the pill recognition challenge through a groundbreaking approach that employs a multi-stream network 
with EfficientNet-B0 and a self-attention mechanism. To eliminate the explicit training of printed or embossed 
patterns, Local Binary Pattern (LBP) features are utilized. Evaluation is performed on two datasets, demonstrating
that our proposed model surpasses previous models in Top-1 and Top-5 accuracy. Notably, the model also outperforms
the YOLOv7 network in a reference-quality use-case.

Branch: `idaacs2023`

## 2. Pill Metrics Learning with Multihead Attention

### Abstract:
In the realm of object recognition, especially where new classes can emerge dynamically, few-shot learning holds
significant importance. Our article focuses on metrics learning, a fundamental technique for few-shot object
recognition, successfully applied to pill recognition. We employ multi-stream metrics learning networks and
explore the integration of multihead attention layers at various points in the network. The model's performance
is evaluated on two datasets, showcasing superior results compared to a state-of-the-art multi-stream pill
recognition network.

Branch: `kdir2023`

## 3. Word and Image Embeddings in Pill Recognition

### Abstract:
Addressing the crucial task of improving pill recognition accuracy within a metrics learning framework, our study
introduces a multi-stream visual feature extraction and processing architecture. Leveraging multi-head attention
layers, we estimate pill similarity. An innovative enhancement to the triplet loss function incorporates word 
embeddings, injecting textual pill similarity into the visual model. This refinement operates on a finer scale 
than conventional triplet loss models, resulting in enhanced visual model accuracy. Experiments and evaluations
are conducted on a new, freely available pill dataset.

Branch: `visapp2024`
