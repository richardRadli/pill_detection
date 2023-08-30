"""
File: split_ogyei_dataset.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023
"""

import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

from tqdm import tqdm
from typing import Dict, List, Tuple

from config.const import DATASET_PATH
from utils.utils import setup_logger


# ------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- G E T   C L A S S E S ---------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def get_classes(images_path) -> \
        Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Get the classes present in the dataset and initialize dictionaries for class counts, train images,
    validation images, and test images.

    :param: Path to the train image files.
    :return: A tuple containing dictionaries for class counts, train images, validation images, and test images.
    """

    classes = set()

    for filename in os.listdir(images_path):
        if filename.endswith('.png'):
            class_name = filename.split('_')[2:-1]
            classes.add('_'.join(class_name))

    classes = sorted(classes)

    class_counts = {class_name: 0 for class_name in classes}
    train_images = {class_name: [] for class_name in classes}
    validation_images = {class_name: [] for class_name in classes}
    test_images = {class_name: [] for class_name in classes}
    return class_counts, train_images, validation_images, test_images


# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- S P L I T   D A T A S E T ------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def split_dataset(class_counts: Dict[str, int], train_images: Dict[str, List[str]],
                  validation_images: Dict[str, List[str]], test_images: Dict[str, List[str]], images_path,
                  valid_split_ratio: float = 0.15, test_split_ratio: float = 0.15) -> \
        Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Split the dataset into train, validation, and test sets based on the class counts.

    :param class_counts: A dictionary containing the counts of each class.
    :param train_images: A dictionary containing the train images for each class.
    :param validation_images: A dictionary containing the validation images for each class.
    :param test_images: A dictionary containing the test images for each class.
    :param images_path: Path to the train images.
    :param valid_split_ratio:
    :param test_split_ratio:
    :return: A tuple containing dictionaries for class counts, train images, validation images, and test images.
    """

    for filename in os.listdir(images_path):
        if filename.endswith('.png'):
            class_name = '_'.join(filename.split('_')[2:-1])
            class_counts[class_name] += 1
            if len(validation_images[class_name]) < round(class_counts[class_name] * valid_split_ratio):
                validation_images[class_name].append(filename)
            elif len(test_images[class_name]) < round(class_counts[class_name] * test_split_ratio):
                test_images[class_name].append(filename)
            else:
                train_images[class_name].append(filename)

    return class_counts, train_images, validation_images, test_images


# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ S T A T   O F   D A T A S E T ------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------- #
def statistics_of_dataset(class_counts: Dict[str, int], train_images: Dict[str, List[str]],
                          validation_images: Dict[str, List[str]], test_images: Dict[str, List[str]]) -> None:
    """
    Calculate and display the statistics of the dataset including image counts per class.

    :param class_counts: A dictionary containing the counts of each class.
    :param train_images: A dictionary containing the train images for each class.
    :param validation_images: A dictionary containing the validation images for each class.
    :param test_images: A dictionary containing the test images for each class.
    :return: None
    """

    logging.info('Image counts per class:')
    results = []
    for class_name, count in class_counts.items():
        train_count = len(train_images[class_name])
        validation_count = len(validation_images[class_name])
        test_count = len(test_images[class_name])
        results.append((class_name, count, train_count, validation_count, test_count))

    df = pd.DataFrame(results, columns=['Class', 'Total', 'Train', 'Validation', 'Test'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df)

    plt.figure(figsize=(10, 6))
    plt.bar(df['Class'], df['Train'], label='Train')
    plt.bar(df['Class'], df['Validation'], bottom=df['Train'], label='Validation')
    plt.bar(df['Class'], df['Test'], bottom=df['Train'] + df['Validation'], label='Test')
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.title('Image counts per class')
    plt.legend()
    plt.tick_params(axis='x', labelrotation=90)
    plt.show()


# ------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- M O V E   F I L E S ----------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def move_files(set_images: Dict[str, List[str]], images_path: str, set_images_path: str, labels_path: str,
               set_labels_path: str, op: str) -> None:
    """
    Move files from the source directory to the destination directory.

    :param set_images: Dictionary containing image filenames for each class.
    :param images_path: Path to the source image directory.
    :param set_images_path: Path to the destination image directory.
    :param labels_path: Path to the source label directory.
    :param set_labels_path: Path to the destination label directory.
    :param op: Operation description for progress display.
    :return: None
    """

    for class_name in tqdm(set_images, desc="Moving %s files" % op):
        for name in set_images[class_name]:
            shutil.copy(os.path.join(images_path, name), os.path.join(set_images_path, name))
            shutil.copy(os.path.join(labels_path, name.replace(".png", ".txt")),
                        os.path.join(set_labels_path, name.replace(".png", ".txt")))


# ------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------- M A I N ----------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def main(replace_files: bool = False) -> None:
    """
    Main function to execute the data processing pipeline.

    :param replace_files: If True, move files to the validation and test directories.
    :return: None
    """

    setup_logger()

    images_path = DATASET_PATH.get_data_path("ogyei_v2_unsplitted_images")
    labels_path = DATASET_PATH.get_data_path("ogyei_v2_unsplitted_labels")

    train_images_path = DATASET_PATH.get_data_path("ogyei_v2_splitted_train_images")
    train_labels_path = DATASET_PATH.get_data_path("ogyei_v2_splitted_train_labels")
    val_images_path = DATASET_PATH.get_data_path("ogyei_v2_splitted_valid_images")
    val_labels_path = DATASET_PATH.get_data_path("ogyei_v2_splitted_valid_labels")
    test_images_path = DATASET_PATH.get_data_path("ogyei_v2_splitted_test_images")
    test_labels_path = DATASET_PATH.get_data_path("ogyei_v2_splitted_test_labels")

    class_counts, train_images, validation_images, test_images = get_classes(images_path)
    class_counts, train_images, validation_images, test_images = split_dataset(class_counts, train_images,
                                                                               validation_images, test_images,
                                                                               images_path)

    statistics_of_dataset(class_counts, train_images, validation_images, test_images)

    if replace_files:
        move_files(set_images=train_images, images_path=images_path, set_images_path=train_images_path,
                   labels_path=labels_path, set_labels_path=train_labels_path, op="train")
        move_files(set_images=validation_images, images_path=images_path, set_images_path=val_images_path,
                   labels_path=labels_path, set_labels_path=val_labels_path, op="validation")
        move_files(set_images=test_images, images_path=images_path, set_images_path=test_images_path,
                   labels_path=labels_path, set_labels_path=test_labels_path, op="test")


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(replace_files=False)
