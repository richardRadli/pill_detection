import cv2
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
import torch

from datetime import datetime
from glob import glob
from os.path import splitext
from PIL import Image
from torch import Tensor
from tqdm import tqdm


from const import CONST


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- P L O T   I M G   &   M A S K -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_img_and_mask(img, mask):
    """

    :param img:
    :param mask:
    :return:
    """

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
    """
    Average of Dice coefficient for all batches, or for a single mask

    :param input_tensor:
    :param target:
    :param reduce_batch_first:
    :param epsilon:
    :return:
    """

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
    """
    Average of Dice coefficient for all classes

    :param input_tensor:
    :param target:
    :param reduce_batch_first:
    :param epsilon:
    :return:
    """

    return dice_coefficient(input_tensor.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ D I C E   L O S S ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def dice_loss(input_tensor: Tensor, target: Tensor, multiclass: bool = False):
    """
    Dice loss (objective to minimize) between 0 and 1

    :param input_tensor:
    :param target:
    :param multiclass:
    :return:
    """

    fn = multiclass_dice_coefficient if multiclass else dice_coefficient
    return 1 - fn(input_tensor, target, reduce_batch_first=True)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ L O A D   I M A G E -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def load_image(filename):
    """

    :param filename:
    :return:
    """

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
    """

    :param idx:
    :param mask_dir:
    :param mask_suffix:
    :return:
    """

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
# ----------------------------------------- R E A D   I M A G E   T O   L I S T ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def read_image_to_list():
    """

    :return:
    """

    images = sorted(glob(CONST.dir_train_images + "*.png"))
    file_names = []
    images_list = []

    for idx, img_path in tqdm(enumerate(images), desc="Reading images", total=len(images)):
        file_names.append(os.path.basename(img_path))
        train_img = cv2.imread(img_path, 1)
        images_list.append(train_img)
    return np.array(images_list), file_names


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
    f, ax = plt.subplots(1, 3)

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    ax[0].imshow(input_image)
    ax[0].set_title("Input")
    ax[1].imshow(gt_image)
    ax[1].set_title("Ground truth")
    ax[2].imshow(pred_image)
    ax[2].set_title("Predicted")

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
# ----------------------- F I N D   L A T E S T   F I L E   I N   L A T E S T   D I R E C T O R Y ----------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_latest_file_in_latest_directory(path: str) -> str:
    """
    Finds the latest file in the latest directory within the given path.

    :param path: str, the path to the directory where we should look for the latest file
    :return: str, the path to the latest file
    :raise: when no directories or files found
    """

    # Get a list of all directories in the given path
    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if not dirs:
        raise ValueError(f"No directories found in {path}")

    # Sort directories by creation time (newest first)
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Get the latest directory
    latest_dir = dirs[0]

    # Get a list of all files in the latest directory
    files = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if
             os.path.isfile(os.path.join(latest_dir, f))]

    if not files:
        raise ValueError(f"No files found in {latest_dir}")

    # Sort files by creation time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Get the latest file
    latest_file = files[0]

    print(f"The latest file is {latest_file}")
    return latest_file


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- F I N D   L A T E S T   F I L E ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_latest_file_in_directory(path: str, extension: str) -> str:
    """
    Finds the latest file in a directory with a given extension.

    :param path: The path to the directory to search for files.
    :param extension: The file extension to look for (e.g. "txt").
    :return: The full path of the latest file with the given extension in the directory.
    """

    files = glob(os.path.join(path, "*.%s" % extension))
    latest_file = max(files, key=os.path.getctime)
    return latest_file


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- P L O T   R E F   Q U E R Y   I M G S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_ref_query_images(indices: list[int], q_images_path: list[str], r_images_path: list[str], gt: list[str],
                          pred_cs: list[str], operation: str) -> None:
    """
    Plots the reference and query images with their corresponding ground truth and predicted class labels.

    :param indices: list of indices representing the matched reference images for each query image
    :param q_images_path: list of file paths to query images
    :param r_images_path: list of file paths to reference images
    :param gt: list of ground truth class labels for each query image
    :param pred_cs: list of predicted class labels for each query image
    :param operation: stream or fusion network
    :return: None
    """

    out_path = CONST.dir_query_ref_pred if operation == "stream" else CONST.dir_fusion_net_pred
    new_list = [i for i in range(len(indices))]

    for idx, (i, j, k, l) in tqdm(enumerate(zip(indices, new_list, gt, pred_cs)), total=len(new_list),
                                  desc="Plotting ref and query images"):
        img_path_query = q_images_path[j]
        img_query = Image.open(img_path_query)

        img_path_ref = r_images_path[int(i)]
        img_ref = Image.open(img_path_ref)

        plt.figure()
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(img_query)
        ax[0].set_title(k + "_query")
        ax[1].imshow(img_ref)
        ax[1].set_title(l + "_ref")

        output_path = os.path.join(out_path, str(idx) + ".png")
        plt.savefig(output_path)
        plt.close()

        plt.close("all")
        plt.close()
        gc.collect()


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ C R E A T E   L A B E L   D I R S -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_label_dirs(rgb_path: str, contour_path: str, texture_path: str) -> None:
    """
    Function create labels. Goes through a directory, yield the name of the medicine(s) from the file name, and create
    a corresponding directory, if that certain directory does not exist. Finally, it copies every image with the same
    label to the corresponding directory.

    :param rgb_path: string, path to the directory.
    :param texture_path:
    :param contour_path:
    :return: None
    """

    files_rgb = os.listdir(rgb_path)
    files_contour = os.listdir(contour_path)
    files_texture = os.listdir(texture_path)

    for idx, (file_rgb, file_contour, file_texture) in tqdm(enumerate(zip(files_rgb, files_contour, files_texture))):
        if file_rgb.endswith(".png"):
            match = re.search(r'^id_\d{3}_([a-zA-Z0-9_]+)_\d{3}\.png$', file_rgb)
            if match:
                value = match.group(1)
                out_path_rgb = os.path.join(rgb_path, value)
                out_path_contour = os.path.join(contour_path, value)
                out_path_texture = os.path.join(texture_path, value)

                os.makedirs(out_path_rgb, exist_ok=True)
                os.makedirs(out_path_contour, exist_ok=True)
                os.makedirs(out_path_texture, exist_ok=True)

                shutil.move(os.path.join(rgb_path, file_rgb), out_path_rgb)
                shutil.move(os.path.join(contour_path, file_contour), out_path_contour)
                shutil.move(os.path.join(texture_path, file_texture), out_path_texture)


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- P R I N T   N E T W O R K   C O N F I G --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def print_network_config(cfg):
    """

    :param cfg:
    :return:
    """

    df = pd.DataFrame.from_dict(vars(cfg), orient='index', columns=['value'])
    print("Parameters of the selected %s\n" % cfg.type_of_net, df)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- F I N D   S T R E A M   F O L D E R S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_stream_folders(path):
    """

    :param path:
    :return:
    """
    found_paths = []

    dirs = sorted(glob(os.path.join(path, '????-??-??_??-??-??')), reverse=True)
    subdir_dict = {'RGB': [], 'Contour': [], 'Texture': []}

    for d in dirs:
        subdirs = ['RGB', 'Contour', 'Texture']
        for subdir in subdirs:
            if os.path.isdir(os.path.join(d, subdir)):
                subdir_dict[subdir].append(d)
                break

        if all(subdir_dict.values()):
            break

    for subdir, dirs in subdir_dict.items():
        print(f"{subdir} directories:")
        for d in dirs:
            print(f"  {d}")
            found_paths.append(d)

    return found_paths


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- P R E D I C T I O N   S T A T I S T I C S -------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def prediction_statistics(stream_network_prediction_file: str, fusion_network_prediction_file: str):
    """

    :param stream_network_prediction_file:
    :param fusion_network_prediction_file:
    :return:
    """

    with open(stream_network_prediction_file, 'r') as f1, open(fusion_network_prediction_file, 'r') as f2:
        f1_lines = f1.readlines()[1:-3]
        f2_lines = f2.readlines()[1:-3]

        differ_list = []

        for line1, line2 in zip(f1_lines, f2_lines):
            cols1 = line1.strip().split('\t')
            cols2 = line2.strip().split('\t')

            if cols1[1] != cols1[2] or cols2[1] != cols2[2]:
                differ_list.append([cols1[1], cols1[2], cols2[2]])

    sn_cnt, fn_cnt, n_count = 0, 0, 0
    sn_list, fn_list, n_list = [], [], []

    for _, (gt, sn, fn) in enumerate(differ_list):
        if gt == sn and gt != fn:
            sn_cnt += 1
            sn_list.append([gt, sn, fn])

        if gt == fn and gt != sn:
            fn_cnt += 1
            fn_list.append([gt, sn, fn])

        if gt != sn and gt != fn:
            n_count += 1
            n_list.append([gt, sn, fn])

    df = pd.DataFrame(differ_list, columns=['Ground Truth', 'StreamNetwork Prediction', 'FusionNetwork Prediction'])
    df_sn = pd.DataFrame(sn_list, columns=['Ground Truth', 'StreamNetwork Prediction', 'FusionNetwork Prediction'])
    df_fn = pd.DataFrame(fn_list, columns=['Ground Truth', 'StreamNetwork Prediction', 'FusionNetwork Prediction'])
    df_n = pd.DataFrame(n_list, columns=['Ground Truth', 'StreamNetwork Prediction', 'FusionNetwork Prediction'])

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("\nMedicines where either StreamNetwork or FusionNetwork predicted well\n", df)
    print("\nMedicines where FusionNetwork predicted wrong and StreamNetwork predicted well\n", df_sn)
    print("\nMedicines where StreamNetwork predicted wrong and FusionNetwork predicted well\n", df_fn)
    print("\nMedicines where neither StreamNetwork nor FusionNetwork predicted well\n", df_n)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- C U D A   I N F O R M A T I O N ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def use_gpu_if_available():
    """
    This function provides information about the currently available GPUs.
    :return: device
    """

    cuda_info = {
        'CUDA Available': [torch.cuda.is_available()],
        'CUDA Device Count': [torch.cuda.device_count()],
        'Current CUDA Device': [torch.cuda.current_device()],
        'CUDA Device Name': [torch.cuda.get_device_name(0)]
    }

    df = pd.DataFrame(cuda_info)
    print("\n", df, "\n")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_json_structure(json_file: str) -> None:
    # Load the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Print the keys of each nested dictionary or element in a list
    for key in data.keys():
        if isinstance(data[key], dict):
            print(f"Keys in '{key}':", data[key].keys())
        elif isinstance(data[key], list):
            print(f"Elements in '{key}':")
            for element in data[key]:
                if isinstance(element, dict):
                    print(element.keys())
        else:
            print(f"Unknown type for '{key}'")
