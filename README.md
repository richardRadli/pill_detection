#Multi-Stream Pill Recognition with Attention
We address the challenge of pill recognition by proposing a novel approach that utilizes a multi-stream network with EfficientNet-B0 and a self-attention mechanism. To get rid of the explicit training of printed or embossed patterns we used LBP features. For evaluation two datasets were utilized. In the test scenarios our proposed model outperformed the previous models in Top-1 and Top-5 accuracy, also the YOLOv7 network in a reference-quality use-case. 

## Datasets

## Requirement
Make sure you have the following dependencies installed:

```bash
torch==2.0.0+cu117
torchsummary==1.5.1
torchvision==0.15.1+cu117
tqdm==4.65.0
```

## Installation
First, clone/download this repository. In the const.py file you will find this:
