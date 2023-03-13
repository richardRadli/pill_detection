import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil
import torch

from datetime import datetime
from glob import glob
from os.path import splitext
from PIL import Image
from torch import Tensor


from const import CONST


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- P L O T   I M G   &   M A S K -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------  D I C E   C O E F F ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ M C L A S S   D I C E   C O E F F -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def multiclass_dice_coefficient(input_tensor: Tensor, target: Tensor, reduce_batch_first: bool = False,
                                epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coefficient(input_tensor.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ D I C E   L O S S ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def dice_loss(input_tensor: Tensor, target: Tensor, multiclass: bool = False):
    """
    Dice loss (objective to minimize) between 0 and 1
    """
    fn = multiclass_dice_coefficient if multiclass else dice_coefficient
    return 1 - fn(input_tensor, target, reduce_batch_first=True)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ L O A D   I M A G E -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ U N I Q U E   M A S K   V A L U E S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ R E A D   I M G   A N D   M A S K -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def read_image_and_mask():
    images = sorted(glob(CONST.dir_train_images + "*.png"))
    masks = sorted(glob(CONST.dir_train_masks + "*.png"))

    images_list, masks_list = [], []

    for idx, (img_path, mask_path) in enumerate(zip(images, masks)):
        train_img = cv2.imread(img_path, 1)
        images_list.append(train_img)
        mask_img = cv2.imread(mask_path, 1)
        masks_list.append(mask_img)

    return np.array(images_list), np.array(masks_list)


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- P L O T   D I A G R A M S ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_diagrams(input_image, gt_image, pred_image, save_path):
    """

    :param input_image:
    :param gt_image:
    :param pred_image:
    :param save_path:
    :return:
    """

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


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- N U M E R I C A L   S O R T --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def numerical_sort(value):
    """
    This function sorts the numerical values correctly.
    :param value: input numbers.
    :return:
    """

    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- C R E A T E   T I M E S T A M P ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_timestamp():
    """
    This function creates a timestamp.
    :return: timestamp
    """

    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- F I N D   L A T E S T   F I L E ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_latest_file(path):
    """
    This function finds the latest file in a directory.
    :param path: input directory to search
    :return:
    """

    latest_dir = None
    latest_dir_time = datetime.fromtimestamp(0)

    # Find the latest directory
    for dir_path, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dirpath = os.path.join(dir_path, dirname)
            modified_time = datetime.fromtimestamp(os.path.getmtime(dirpath))
            if modified_time > latest_dir_time:
                latest_dir = dirpath
                latest_dir_time = modified_time

    latest_file = None
    if latest_dir is not None:
        latest_file_time = datetime.fromtimestamp(0)
        for filename in os.listdir(latest_dir):
            filepath = os.path.join(latest_dir, filename)
            if os.path.isfile(filepath):
                modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if modified_time > latest_file_time:
                    latest_file = filepath
                    latest_file_time = modified_time

        if latest_file is not None:
            print(f"The latest file is {latest_file}")
        else:
            print("No files found in the latest directory")
    else:
        print("No directories found in the path")

    return latest_file


def copy_ref_images(source_directory: str, destination_directory: str):
    for subdir_name in os.listdir(source_directory):
        subdir_path = os.path.join(source_directory, subdir_name)

        if os.path.isdir(subdir_path):
            file_to_pick = os.listdir(subdir_path)[0]
            latest_file_path = os.path.join(subdir_path, file_to_pick)
            destination_file_path = os.path.join(destination_directory, "tex", latest_file_path.split("\\")[2])
            shutil.copy2(latest_file_path, destination_file_path)


def segment_pills(img_path, img_gray_path, label, predicted_value):
    img = cv2.imread(img_path, 1)
    img_gray = cv2.imread(img_gray_path, 0)

    # Create a blank 3 channel image with zeros
    img_color = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)

    # Merge the gray image with the blank color image to get a 3 channel image
    img_color = cv2.merge((img_gray, img_color[:, :, 1], img_color[:, :, 2]))
    vis_img = cv2.addWeighted(img, 1.0, img_color, 0.3, 0)

    cv2.putText(vis_img, str(label) + ": " + str(np.round(predicted_value, 4)), (0, 50),
                cv2.FONT_ITALIC, 1, (255, 0, 0))

    cv2.imshow("vis_img", vis_img)
    cv2.waitKey()


def plot_ref_query_imgs(indecies, q_images_path, r_images_path, gt, pred_cs):
    new_list = [i for i in range(len(indecies))]

    for idx, (i, j, k, l) in enumerate(zip(indecies, new_list, gt, pred_cs)):
        img_path_query = q_images_path[j]
        img_query = Image.open(img_path_query)

        img_path_ref = r_images_path[int(i)]
        img_ref = Image.open(img_path_ref)

        plt.figure()
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img_query)
        axarr[0].set_title(k + "_query")
        axarr[1].imshow(img_ref)
        axarr[1].set_title(l + "_ref")

        output_path = (CONST.dir_query_ref_pred + "/" + str(idx) + ".png")
        plt.savefig(output_path)
        plt.close()

        plt.close("all")
        plt.close()
        gc.collect()
