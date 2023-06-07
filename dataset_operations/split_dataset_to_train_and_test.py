"""
File: split_dataset_to_train_and_test.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: This program is used to split a dataset into train and test sets. It reads the dataset files, calculates
the class counts, and then splits the files into train and test sets based on the class counts. It also provides
statistics about the dataset and can move the files to the test directory if specified.
"""

import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

from tqdm import tqdm
from typing import Dict, List, Tuple

from config.const import IMAGES_PATH
from config.logger_setup import setup_logger


# ------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- G E T   C L A S S E S ---------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def get_classes(train_images_path) -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Get the classes present in the dataset and initialize dictionaries for class counts, train images, and test images.

    :param: Path to the train image files.
    :return: A tuple containing dictionaries for class counts, train images, and test images.
    """

    classes = set()

    for filename in os.listdir(train_images_path):
        if filename.endswith('.png'):
            class_name = filename.split('_')[2:-1]
            classes.add('_'.join(class_name))

    classes = sorted(classes)

    class_counts = {class_name: 0 for class_name in classes}
    train_images = {class_name: [] for class_name in classes}
    test_images = {class_name: [] for class_name in classes}
    return class_counts, train_images, test_images


# ------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------- S P L I T   D A T A S E T -------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def split_dataset(class_counts: Dict[str, int], train_images: Dict[str, List[str]], test_images: Dict[str, List[str]],
                  train_images_path) -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Split the dataset into train and test sets based on the class counts.

    :param class_counts: A dictionary containing the counts of each class.
    :param train_images: A dictionary containing the train images for each class.
    :param test_images: A dictionary containing the test images for each class.
    :param train_images_path: Path to the train images.
    :return: A tuple containing dictionaries for class counts, train images, and test images.
    """

    for filename in os.listdir(train_images_path):
        if filename.endswith('.png'):
            class_name = '_'.join(filename.split('_')[2:-1])
            class_counts[class_name] += 1
            if len(test_images[class_name]) < round(class_counts[class_name] * 0.2):
                test_images[class_name].append(filename)
            else:
                train_images[class_name].append(filename)

    return class_counts, train_images, test_images


# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ S T A T   O F   D A T A S E T ------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------- #
def statistics_of_dataset(class_counts: Dict[str, int], train_images: Dict[str, List[str]],
                          test_images: Dict[str, List[str]]) -> None:
    """
    Calculate and display the statistics of the dataset including image counts per class.

    :param class_counts: A dictionary containing the counts of each class.
    :param train_images: A dictionary containing the train images for each class.
    :param test_images: A dictionary containing the test images for each class.
    :return: None
    """

    logging.info('Image counts per class:')
    results = []
    for class_name, count in class_counts.items():
        train_count = len(train_images[class_name])
        test_count = len(test_images[class_name])
        results.append((class_name, count, train_count, test_count))

    df = pd.DataFrame(results, columns=['Class', 'Total', 'Train', 'Test'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    logging.info(df)

    plt.figure(figsize=(10, 6))
    plt.bar(df['Class'], df['Train'], label='Train')
    plt.bar(df['Class'], df['Test'], bottom=df['Train'], label='Test')
    plt.xlabel('Class')
    plt.ylabel('Number of images')
    plt.title('Image counts per class')
    plt.legend()
    plt.tick_params(axis='x', labelrotation=90)
    plt.show()


# ------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- M O V E   F I L E S ----------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def move_files(test_images: Dict[str, List[str]], train_images_path, test_images_path, train_masks_path,
               test_masks_path) -> None:
    """
    Move the files from the train directories to the test directories based on the test images.

    :param test_images: A dictionary containing the test images for each class.
    :param train_images_path: Path to the train images.
    :param test_images_path: Path to the test images.
    :param train_masks_path: Path to the train mask images.
    :param test_masks_path: Path to the test mask images.
    :return: None
    """

    for class_name in tqdm(test_images, desc="Moving files"):
        for name in test_images[class_name]:
            shutil.move(os.path.join(train_images_path, name), os.path.join(test_images_path, name))
            shutil.move(os.path.join(train_masks_path, name), os.path.join(test_masks_path, name))


# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------- R O L L B A C K ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def rollback_files(src_dir: str, dst_dir: str) -> None:
    """
    Rollback the files from the destination directory to the source directory.

    :param src_dir: The source directory.
    :param dst_dir: The destination directory.
    :return: None
    """

    for file_name in tqdm(os.listdir(dst_dir), total=len(os.listdir(dst_dir)), desc="Rolling back files"):
        src_path = os.path.join(dst_dir, file_name)
        dst_path = os.path.join(src_dir, file_name)
        shutil.move(src_path, dst_path)


# ------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------- M A I N ----------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def main(replace_files: bool = False, rollback: bool = False) -> None:
    """
    Main function to execute the data processing pipeline.

    :param replace_files: If True, move files to the test directory.
    :param rollback: If True, rollback the files from the test directory to the train directory.
    :return: None
    """

    setup_logger()

    train_images_path = IMAGES_PATH.get_data_path("train_images")
    test_images_path = IMAGES_PATH.get_data_path("test_images")
    train_masks_path = IMAGES_PATH.get_data_path("train_masks")
    test_masks_path = IMAGES_PATH.get_data_path("test_masks")

    class_counts, train_images, test_images = get_classes(train_images_path)
    class_counts, train_images, test_images = split_dataset(class_counts, train_images, test_images, train_images_path)

    statistics_of_dataset(class_counts, train_images, test_images)

    if replace_files:
        move_files(test_images, train_images_path=train_images_path, test_images_path=test_images_path,
                   train_masks_path=train_masks_path, test_masks_path=test_masks_path)

    if rollback:
        rollback_files(train_images_path, test_images_path)
        rollback_files(train_masks_path, test_masks_path)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------- __M A I N__ ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(replace_files=True, rollback=False)
