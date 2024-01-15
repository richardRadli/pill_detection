# Multi-Stream Pill Recognition with Attention
We address the challenge of pill recognition by proposing a novel approach that utilizes a multi-stream network 
with EfficientNet-B0 and a self-attention mechanism. To get rid of the explicit training of printed or embossed 
patterns we used LBP features. For evaluation two datasets were utilized. In the test scenarios our proposed 
model outperformed the previous models in Top-1 and Top-5 accuracy, also the YOLOv7 network in a reference-quality 
use-case. 

## Datasets
We used two datasets, namely CURE[1] and our novel, custom-made one, entitled OGYEIv1. CURE is available online via this
link:

https://drive.google.com/drive/folders/1dcqUaTSepplc4GAUC05mr9iReWVqaThN.

Ours can be accessed if you contact me via my e-mail address: radli.richard@mik.uni-pannon.hu

The comparison of the two datasets can be seen in the table below:

|                        | CURE                | OGYEIv1   |
|------------------------|---------------------|-----------|
| Number of pill classes | 196                 | 78        |
| Number of images       | 8973                | 3154      |
| Image resolution       | 800×800 - 2448×2448 | 2465×1683 |
| Instance per class     | 40-50               | 40-60     |
| Segmentation labels    | no                  | fully     |
| Backgrounds            | 6                   | 1         | 
| Imprinted text labels  | yes                 | yes       |

## Requirement
Make sure you have the following dependencies installed:

```bash
colorama>=0.4.6
colorlog>=6.7.0
json>=2.0.9
matplotlib>=3.7.1
numpy>=1.23.5
opencv-python>=4.5.5.64
pandas>=2.0.0
Pillow>=9.3.0
seaborn>=0.12.2
segmentation_models_pytorch>=0.3.3
skimage>=0.20.0
sklearn>=1.2.2
tkinter>=8.6.12
torch>=2.0.0+cu117
torchsummary>=1.5.1
torchvision>=0.15.1+cu117
tqdm>=4.65.0
```

## Installation
First, clone/download this repository. In the const.py file you will find this:

```python
root_mapping = {
    'ricsi': {
        "PROJECT_ROOT": 'D:/pill_detect/storage/',
        "DATASET_ROOT": 'D:/pill_detect/datasets'
    }
}
```

- Update the designated username ('ricsi') to reflect the username associated with your logged-in operating system.
- Utilize PROJECT_ROOT as the central repository for storing essential data.
- Employ DATASET_ROOT as the designated directory for managing datasets integral to the functioning of the project.
- const.py will create all the necessary folders.
- Download the datasets and place them into the appropriate folders.


## Overview of the repository
In the config.py file, key parameters and settings crucial for the training, testing, and data augmentation processes 
are centrally stored. These configurations provide a streamlined and organized approach to manage various aspects 
of the project, ensuring adaptability and ease of customization.

To create a dataset, we attached a couple of scripts for image capturing. Steps to create a dataset:

- Use `take_calibration_images.py`, and capture images for camera calibration. The more images are taken, 
better the results will be.
- Use `camera_calibration.py` to calibrate the camera.
- Use `camera_recording.py`, and create the image dataset.
- Finally, use `undistort_images.py` to undistort the taken images.

A couple of useful scripts are also part of this repository, these files can be found in the **dataset_operations** folder.
These tools are handful, if you wish to make operations on the datasets, such as splitting, checking the balance and
annotations, etc. Also, it is worth pointing out to the `utils.py` script, that has many useful functions.

## Usage

If the repository is cloned/downloaded, the root paths are sorted out, the datasets are in place, and everything is 
set up in the config files, the next step is to train the UNet, in order to detect pills on the images. For this, use the 
`train_unet.py` file. After training, you can test the ability of the network, but most importantly, make predictions
on the images. These images will be used later on.

Alternatively, if you can use `draw_masks.py` to create the binary mask images.

To create the images for the streams, run `create_stream_images.py`. Make sure you go through all the choices of 
the argument called **dataset_operation** in the **ConfigStreamNetwork** class in the `config.py` file (train, valid, test).

Next step is to train the stream networks, this is Phase 1. 
There are two kind of backbones are available for this, the one published by 
Ling et al. [1] and EfficientNet V1 b0 [2]. Make sure you train all four streams. Also, two loss functions are provided:
triplet loss and hard mining triplet loss. The later will save the hardest triplets, which can be utilized in Phase 2.
To copy the hardest samples into a directory, run `mine_hard_samples.py`.

After the streams are trained, the last step is to train the fusion network, it is also called Phase 2.
Still, there are two choices here: CNN or EfficientNet, but only one loss function: triplet loss. If hard mining triplet
loss was selected in Phase 1, the network will be trained on only the hardest samples.

To evaluate the models, use `predict_fusion_network.py`

## References