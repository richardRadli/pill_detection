import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
import re
import torch

from const import CONST
from glob import glob
from os.path import splitext
from PIL import Image
from torch import Tensor


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def dice_coefficient(input_tensor: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input_tensor.size() == target.size()
    assert input_tensor.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input_tensor.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input_tensor * target).sum(dim=sum_dim)
    sets_sum = input_tensor.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coefficient(input_tensor: Tensor, target: Tensor, reduce_batch_first: bool = False,
                                epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coefficient(input_tensor.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input_tensor: Tensor, target: Tensor, multiclass: bool = False):
    """
    Dice loss (objective to minimize) between 0 and 1
    """
    fn = multiclass_dice_coefficient if multiclass else dice_coefficient
    return 1 - fn(input_tensor, target, reduce_batch_first=True)


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def read_image_and_mask():
    images = sorted(glob(CONST.dir_img + "*.png"))
    masks = sorted(glob(CONST.dir_mask + "*.png"))

    images_list, masks_list = [], []

    for idx, (img_path, mask_path) in enumerate(zip(images, masks)):
        train_img = cv2.imread(img_path, 1)
        images_list.append(train_img)
        mask_img = cv2.imread(mask_path, 1)
        masks_list.append(mask_img)

    return np.array(images_list), np.array(masks_list)


def plot_diagrams(input_image, gt_image, pred_image, save_path):
    plt.figure()

    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1, 3)

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(input_image)
    axarr[0].set_title("Input")
    axarr[1].imshow(gt_image)
    axarr[1].set_title("Ground truth")
    axarr[2].imshow(pred_image)
    axarr[2].set_title("Predicted")

    plt.savefig(save_path)
    plt.close()

    plt.close("all")
    plt.close()
    gc.collect()


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
