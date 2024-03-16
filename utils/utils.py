"""
File: utils.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: This code holds different functions used all around the project files.
"""

import colorlog
import gc
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import time
import torch

from datetime import datetime
from functools import wraps
from glob import glob
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, random_split
from typing import Any, Callable, List, Optional, Tuple, Union
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- C R E A T E   D A T A S E T --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_dataset(dataset, train_valid_ratio: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the dataset into training and validation sets, creates data loaders, and logs information.

    Args:
        dataset: The dataset to be split.
        train_valid_ratio (float): The ratio of the dataset to be used for training.
        batch_size (int): The batch size for the data loaders.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and validation data loaders.

    """

    train_size = int(len(dataset) * train_valid_ratio)
    val_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])
    logging.info(f"Number of images in the train set: {len(train_dataset)}")
    logging.info(f"Number of images in the validation set: {len(valid_dataset)}")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_data_loader, valid_data_loader


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- C R E A T E   T I M E S T A M P ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_timestamp() -> str:
    """
    Creates a timestamp string representing the current date and time.

    Returns:
        str: The timestamp string formatted as '%Y-%m-%d_%H-%M-%S'.
    """

    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ F I L E   R E A D E R -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def file_reader(file_path: str, extension: str) -> List[str]:
    """
    Reads files with a specific extension from a given directory and sorts them numerically.

    Args:
        file_path (str): The path to the directory containing the files.
        extension (str): The extension of the files to be read.

    Returns:
        List[str]: A sorted list of filenames with the specified extension.
    """

    return sorted([str(file) for file in Path(file_path).glob(f'*.{extension}')], key=numerical_sort)


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- F I N D   L A T E S T   D I R E C T O R Y -------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_latest_directory(path: str):
    """

    Args:
        path:

    Return:

    """
    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if not dirs:
        raise ValueError(f"No directories found in {path}")

    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return dirs[0]


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- F I N D   L A T E S T   F I L E ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_latest_file_in_directory(path: str, extension: str) -> str:
    """
    Finds the latest file in a directory with a given extension.

    Args:
        path (str): The path to the directory to search for files.
        extension (str): The file extension to look for (e.g., "txt").

    Returns:
        str: The full path of the latest file with the given extension in the directory.
    """

    files = glob(os.path.join(path, "*.%s" % extension))
    latest_file = max(files, key=os.path.getctime)
    return latest_file


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------- F I N D   L A T E S T   F I L E   I N   L A T E S T   D I R E C T O R Y ----------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_latest_file_in_latest_directory(path: str, type_of_loss: str = None) -> str:
    """
    Finds the latest file in the latest directory within the given path.

    Args:
        path (str): The path to the directory where we should look for the latest file.
        type_of_loss (str, optional): The type of loss to filter directories. Defaults to None.

    Returns:
        str: The path to the latest file.

    Raises:
        ValueError: When no directories or files are found.
    """

    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if len(dirs) == 0:
        raise ValueError(f"No directories found in {path}")

    dirs = [path for path in dirs if type_of_loss in path.lower()]
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = dirs[0]
    files = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if
             os.path.isfile(os.path.join(latest_dir, f))]

    if not files:
        raise ValueError(f"No files found in {latest_dir}")

    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = files[0]
    logging.info(f"The latest file is {latest_file}")

    return latest_file


def find_latest_file_in_latest_directory_word_emb(path) -> str:
    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if len(dirs) == 0:
        raise ValueError(f"No directories found in {path}")

    dirs = [path for path in dirs]
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_dir = dirs[0]
    files = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if
             os.path.isfile(os.path.join(latest_dir, f))]

    if not files:
        raise ValueError(f"No files found in {latest_dir}")

    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = files[0]
    logging.info(f"The latest file is {latest_file}")

    return latest_file

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- F I N D   L A T E S T   S U B D ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_latest_subdir(directory: str) -> Optional[str]:
    """
    Finds the latest subdirectory within the given directory.

    Args:
        directory (str): The path to the directory where we should look for the latest subdirectory.

    Returns:
        str or None: The path to the latest subdirectory if found, otherwise None.
    """

    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    if not subdirs:
        print(f"No subdirectories found in {directory}.")
        return None

    latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(directory, d)))

    return os.path.join(directory, latest_subdir)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------- G E T   E M B E D D E D   T E X T   M A T R I X ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_embedded_text_matrix(path_to_excel_file: str) -> pd.DataFrame:
    """
    Reads an Excel file and returns the data from the first sheet as a DataFrame.

    Args:
        path_to_excel_file (str): The path to the directory containing the Excel file.

    Returns:
        pd.DataFrame: The data from the first sheet of the Excel file as a DataFrame.

    Raises:
        ValueError: If the Excel file does not exist at the specified path.
    """

    excel_file_path = find_latest_file_in_directory(path_to_excel_file, "xlsx")
    if not os.path.exists(excel_file_path):
        raise ValueError(f"Excel file at path {excel_file_path} doesn't exist")
    return pd.read_excel(excel_file_path, sheet_name=0, index_col=0)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- M E A S U R E   E X E C U T I O N   T I M E ------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        """
        Wrapper function to measure execution time.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Any: The result of the function.
        """

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result

    return wrapper


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- M I N E   H A R D   T R I P L E T S  ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def mine_hard_triplets(latest_txt: str) -> List[Tuple[str, str, str]]:
    """
    Mines hard triplets from the data processed in the latest text file.

    Args:
        latest_txt (str): Path to the latest text file containing processed data.

    Returns:
        List[Tuple[str, str, str]]: List of hard triplets, where each triplet consists of an anchor, positive, and
        negative sample.
    """

    hardest_samples = process_txt(latest_txt)
    triplets = []
    for i, samples in enumerate(hardest_samples):
        for a, p, n in zip(samples[0], samples[1], samples[2]):
            triplets.append((a, p, n))
    return list(set(triplets))


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- N U M E R I C A L   S O R T --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def numerical_sort(value: str) -> List[Union[str, int]]:
    """
    Sorts numerical values in a string ensuring correct numerical sorting.

    Args:
        value (str): The input string containing numerical and non-numerical parts.

    Returns:
        List[Union[str, int]]: A list containing both strings and integers sorted by numerical value.
    """

    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def plot_confusion_matrix(gt: List[str], predictions: List[str], out_path: str) -> None:
    """
    Plots and saves the confusion matrix based on ground truth and predicted labels.

    Args:
        gt (List[str]): Ground truth labels.
        predictions (List[str]): Predicted labels.
        out_path (str): Path to the output directory to save the plot.

    Returns:
        None
    """

    # Get unique labels from ground truth and predictions
    labels = list(set(gt + predictions))

    # Create a mapping from labels to unique integers
    label_to_int = {label: i for i, label in enumerate(labels)}

    # Convert labels to integers using the mapping
    true_labels = [label_to_int[label] for label in gt]
    predicted_labels = [label_to_int[label] for label in predictions]

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Sort labels based on unique integers
    sorted_labels = [label for label, _ in sorted(label_to_int.items(), key=lambda x: x[1])]

    # Create a heatmap
    plt.figure(figsize=(20, 12))
    sns.set(font_scale=1.2)  # Adjust the font size for better readability
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted_labels, yticklabels=sorted_labels)

    # Add labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Prevent label cutoff
    plt.tight_layout()

    # Show the plot
    os.makedirs(out_path, exist_ok=True)
    output_path = os.path.join(out_path, "confusion_matrix.png")
    plt.savefig(output_path, dpi=600)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- P L O T   R E F   Q U E R Y   I M G S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_ref_query_images(indices: List[int], q_images_path: List[str], r_images_path: List[str],
                          gt: List[str], predicted_labels: List[str], output_folder: str) -> None:
    """
    Plots the reference and query images with their corresponding ground truth and predicted class labels.

    Args:
        indices (List[int]): List of indices representing the matched reference images for each query image.
        q_images_path (List[str]): List of file paths to query images.
        r_images_path (List[str]): List of file paths to reference images.
        gt (List[str]): List of ground truth class labels for each query image.
        predicted_labels (List[str]): List of predicted class labels for each query image.
        output_folder (str): Path to the output folder where the images will be saved.

    Returns:
        None
    """

    new_list = [i for i in range(len(indices))]

    correctly_classified = os.path.join(output_folder, "correctly_classified")
    incorrectly_classified = os.path.join(output_folder, "incorrectly_classified")

    os.makedirs(correctly_classified, exist_ok=True)
    os.makedirs(incorrectly_classified, exist_ok=True)

    for idx, (i, j, k, l) in tqdm(enumerate(zip(indices, new_list, gt, predicted_labels)), total=len(new_list),
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

        if k == l:
            output_path = os.path.join(correctly_classified, str(idx) + ".png")
        else:
            output_path = os.path.join(incorrectly_classified, str(idx) + ".png")
        plt.savefig(output_path)
        plt.close()

        plt.close("all")
        plt.close()
        gc.collect()


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- P R I N T   N E T W O R K   C O N F I G --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def print_network_config(cfg) -> None:
    """
    Prints the network configuration parameters.

    Args:
        cfg (object): The configuration object containing network parameters.

    Returns:
        None
    """

    df = pd.DataFrame.from_dict(vars(cfg), orient='index', columns=['value'])
    logging.info("Parameters of the selected %s:", cfg.type_of_net)
    logging.info(df)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ P R O C E S S   T X T -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def process_txt(txt_file: str) -> list:
    """
    Reads a .txt file and extracts a set of paths from its contents.

    Args:
        txt_file (str): The path to the .txt file.

    Returns:
        list: A list of paths extracted from the .txt file.
    """

    paths = []

    with open(txt_file, 'r') as f:
        data = eval(f.read())

    for key in data:
        paths.append(key)

    return paths


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- S E T U P   L O G G E R ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def setup_logger():
    """
    Set up a colorized logger with the following log levels and colors:

    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red on a white background

    Returns:
        The configured logger instance.
    """

    # Check if logger has already been set up
    logger = logging.getLogger()
    if logger.hasHandlers():
        return logger

    # Set up logging
    logger.setLevel(logging.INFO)

    # Create a colorized formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        })

    # Create a console handler and add the formatter to it
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- S O R T   D I R E C T O R Y ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def sort_dict(directory):
    """

    Args:
        directory:

    Returns:

    """

    if all(key.isdigit() for key in directory.keys()):
        sorted_dict = dict(sorted(directory.items(), key=lambda item: int(item[0])))
    else:
        sorted_tuples = sorted(directory.items())
        sorted_dict = dict(sorted_tuples)

    return sorted_dict


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- U S E   G P U   I F   A V A I L A B L E --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def use_gpu_if_available() -> torch.device:
    """
    Provides information about the currently available GPUs and returns a torch device for training and inference.

    Returns:
        torch.device: A torch device for either "cuda" or "cpu".
    """

    if torch.cuda.is_available():
        cuda_info = {
            'CUDA Available': [torch.cuda.is_available()],
            'CUDA Device Count': [torch.cuda.device_count()],
            'Current CUDA Device': [torch.cuda.current_device()],
            'CUDA Device Name': [torch.cuda.get_device_name(0)]
        }

        df = pd.DataFrame(cuda_info)
        logging.info(df)
    else:
        logging.info("Only CPU is available!")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
