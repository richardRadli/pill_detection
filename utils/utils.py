"""
File: utils.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: This code holds different function used all around the project files.
"""

import colorlog
import gc
import json
import jsonschema
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import time
import torch

from datetime import datetime
from functools import wraps
from jsonschema import validate
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from typing import List, Union, Tuple


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
# ------------------------------------------------ F I L E   R E A D E R -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def file_reader(file_path: str, extension: str) -> List[str]:
    """
    Args:
        file_path:
        extension:

    Returns:

    """

    return sorted([str(file) for file in Path(file_path).glob(f'*.{extension}')], key=numerical_sort)


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- N U M E R I C A L   S O R T --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def numerical_sort(value: str) -> List[Union[str, int]]:
    """
    Sort numerical values in a string in a way that ensures numerical values are sorted correctly.

        value: The input string.
    Returns: A list of strings and integers sorted by numerical value.
    """

    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- C R E A T E   T I M E S T A M P ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_timestamp() -> str:
    """
    Creates a timestamp in the format of '%Y-%m-%d_%H-%M-%S', representing the current date and time.

    Returns: The timestamp string.
    """

    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------- F I N D   L A T E S T   F I L E   I N   L A T E S T   D I R E C T O R Y ----------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_latest_file_in_latest_directory(path: str) -> str:
    """
    Finds the latest file in the latest directory within the given path.

    Args:
        path (str): The path to the directory where we should look for the latest file.

    Returns:
        str: The path to the latest file.

    Raises:
        ValueError: When no directories or files are found.
    """

    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if len(dirs) == 0:
        raise ValueError(f"No directories found in {path}")

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
# ---------------------------------------- P L O T   R E F   Q U E R Y   I M G S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_ref_query_images(gt_labels: List[str], predicted_medicines: List[str],
                          query_image_paths: dict, reference_image_paths: dict,
                          output_folder: str) -> None:
    """

    Args:
        gt_labels: 
        predicted_medicines: 
        query_image_paths: 
        reference_image_paths: 
        output_folder: 

    Returns:

    """

    correctly_classified = os.path.join(output_folder, "correctly_classified")
    incorrectly_classified = os.path.join(output_folder, "incorrectly_classified")
    os.makedirs(correctly_classified, exist_ok=True)
    os.makedirs(incorrectly_classified, exist_ok=True)

    # Loop through ground truth and predicted labels
    for i, (gt_label, predicted_medicine) in tqdm(enumerate(zip(gt_labels, predicted_medicines)),
                                                  total=len(gt_labels),
                                                  desc="Plotting images"):

        query_images = query_image_paths[predicted_medicine]

        if gt_label == predicted_medicine:
            save_folder = correctly_classified
            ref_label = gt_label
        else:
            save_folder = incorrectly_classified
            ref_label = gt_label

        reference_images = [Image.open(path) for path in reference_image_paths[ref_label]]

        query_image = Image.open(
            query_images[i % len(query_images)])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for j, ref_img in enumerate(reference_images):
            axes[j].imshow(ref_img)
            axes[j].axis('off')
            axes[j].set_title(f'Reference Image {j + 1} ({ref_label})')

        axes[2].imshow(query_image)
        axes[2].axis('off')
        axes[2].set_title(f'Query Image ({predicted_medicine})')

        plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.2, wspace=0.3)

        plot_filename = os.path.join(save_folder, f'{gt_label}_vs_{predicted_medicine}_query_{i}.png')
        plt.savefig(plot_filename, dpi=100)
        plt.close()

    gc.collect()


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- U S E   G P U   I F   A V A I L A B L E --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def use_gpu_if_available() -> torch.device:
    """
    Provides information about the currently available GPUs and returns a torch device for training and inference.

    Returns: A torch device for either "cuda" or "cpu".
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


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- M E A S U R E   E X E C U T I O N   T I M E ------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def measure_execution_time(func):
    """
    Decorator to measure the execution time.

        func:
    Returns:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ P R O C E S S   T X T -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def process_txt(txt_file: str) -> list:
    """
    Reads a .txt file and extracts a set of paths from its contents.

        txt_file: The path to the .txt file.

    Returns:
        A set of paths extracted from the .txt file.
    """

    paths = []

    with open(txt_file, 'r') as f:
        data = eval(f.read())

    for key in data:
        paths.append(key)

    return paths


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- M I N E   H A R D   T R I P L E T S ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def mine_hard_triplets(latest_txt):
    """

        latest_txt:

    Returns:

    """

    hardest_samples = process_txt(latest_txt)
    triplets = []
    for i, samples in enumerate(hardest_samples):
        for a, p, n in zip(samples[0], samples[1], samples[2]):
            triplets.append((a, p, n))
    return list(set(triplets))


def load_config_json(json_schema_filename: str, json_filename: str):
    """
    Args:
        json_schema_filename:
        json_filename:

    Returns:

    """

    with open(json_schema_filename, "r") as schema_file:
        schema = json.load(schema_file)

    with open(json_filename, "r") as config_file:
        config = json.load(config_file)

    try:
        validate(config, schema)
        logging.info("JSON data is valid.")
        return config
    except jsonschema.exceptions.ValidationError as err:
        logging.error(f"JSON data is invalid: {err}")


def print_settings(file_name):
    with open(file_name, "r") as f:
        json_file = json.load(f)

    for k, v in json_file.items():
        if k not in ["networks", "streams"]:
            logging.info(f"{k}: {v}")

    type_of_net = json_file.get("type_of_net")
    if type_of_net and type_of_net in json_file.get("networks", {}):
        logging.info(f"\nNetwork ({type_of_net}) configuration:")
        for k, v in json_file["networks"][type_of_net].items():
            logging.info(f"  {k}: {v}")
    else:
        logging.info(f"\nNo configuration found for network type: {type_of_net}")

    type_of_stream = json_file.get("type_of_stream")
    if type_of_stream and type_of_stream in json_file.get("streams", {}):
        logging.info(f"\nStream ({type_of_stream}) configuration:")
        for k, v in json_file["streams"][type_of_stream].items():
            logging.info(f"  {k}: {v}")
    else:
        logging.info(f"\nNo configuration found for stream type: {type_of_stream}")
