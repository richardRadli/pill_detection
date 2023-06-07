"""
File: split_ogyi_dataset.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: May 03, 2023

Description: This program is used to split a dataset into train, validation, and test sets. It collects the data from a
source directory and organizes it into train, validation, and test directories. The image files are shuffled and
distributed among the directories accordingly.
"""

import os
import shutil

from random import sample
from tqdm import tqdm

from config.const import DATASET_PATH


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- S O R T   F I L E S ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def sort_files() -> None:
    """
    Sorts the image files into train, valid, and test sets and copies them to the respective directories.
    :return: None
    """

    source_dir = '/'.join(DATASET_PATH.get_data_path("ogyi_v2_unsplitted_images").split('/')[0:-1])
    train_dir = '/'.join(DATASET_PATH.get_data_path("ogyi_v2_splitted_train_images").split('/')[0:-1])
    valid_dir = '/'.join(DATASET_PATH.get_data_path("ogyi_v2_splitted_valid_images").split('/')[0:-1])
    test_dir = '/'.join(DATASET_PATH.get_data_path("ogyi_v2_splitted_test_images").split('/')[0:-1])

    image_files = os.listdir()

    # Shuffle the image files
    sampled_files = sample(image_files, len(image_files))

    train_image_files = sampled_files[:int(0.7 * len(image_files))]
    valid_image_files = sampled_files[int(0.7 * len(image_files)):int(0.85 * len(image_files))]
    test_image_files = sampled_files[int(0.85 * len(image_files)):]

    for file in tqdm(train_image_files, total=len(train_image_files), desc="Train images"):
        copy_files(train_dir, file, source_dir)

    for file in tqdm(valid_image_files, total=len(valid_image_files), desc="Valid images"):
        copy_files(valid_dir, file, source_dir)

    for file in tqdm(test_image_files, total=len(test_image_files), desc="Testing images"):
        copy_files(test_dir, file, source_dir)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- C O P Y   F I L E S ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def copy_files(dir_type: str, file: str, source_dir: str) -> None:
    """
    Copies the image and corresponding text files to the target directory.

    :param dir_type: The type of directory (e.g., train, valid, test).
    :param file: The name of the file to be copied.
    :param source_dir: The source directory containing the files.
    :return: None
    """

    # Copy the image file to the train/images directory
    image_source_file = os.path.join(source_dir, "images", file)
    image_target_file = os.path.join(dir_type, "images", file)
    shutil.copyfile(image_source_file, image_target_file)

    # Copy the corresponding text file to the train/labels directory
    text_file = os.path.splitext(file)[0] + ".txt"
    text_source_file = os.path.join(source_dir, "labels", text_file)
    text_target_file = os.path.join(dir_type, "labels", text_file)
    shutil.copyfile(text_source_file, text_target_file)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- M A I N ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    """

    :return: None
    """

    sort_files()


if __name__ == "__main__":
    main()
