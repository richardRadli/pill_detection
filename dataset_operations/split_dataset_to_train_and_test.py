import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

from tqdm import tqdm
from typing import Dict, List, Tuple

from config.const import CONST
from config.logger_setup import setup_logger


# ------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- G E T   C L A S S E S ---------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def get_classes() -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Get the classes present in the dataset and initialize dictionaries for class counts, train images, and test images.

    :return: A tuple containing dictionaries for class counts, train images, and test images.
    """

    classes = set()

    for filename in os.listdir(CONST.dir_train_images):
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
def split_dataset(class_counts: Dict[str, int], train_images: Dict[str, List[str]], test_images: Dict[str, List[str]]) \
        -> Tuple[Dict[str, int], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Split the dataset into train and test sets based on the class counts.

    :param class_counts: A dictionary containing the counts of each class.
    :param train_images: A dictionary containing the train images for each class.
    :param test_images: A dictionary containing the test images for each class.
    :return: A tuple containing dictionaries for class counts, train images, and test images.
    """

    for filename in os.listdir(CONST.dir_train_images):
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
def move_files(test_images: Dict[str, List[str]]) -> None:
    """
    Move the files from the train directories to the test directories based on the test images.

    :param test_images: A dictionary containing the test images for each class.
    :return: None
    """

    for class_name in tqdm(test_images, desc="Moving files"):
        for name in test_images[class_name]:
            shutil.move(os.path.join(CONST.dir_train_images, name), os.path.join(CONST.dir_test_images, name))
            shutil.move(os.path.join(CONST.dir_train_masks, name), os.path.join(CONST.dir_test_mask, name))


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

    class_counts, train_images, test_images = get_classes()
    class_counts, train_images, test_images = split_dataset(class_counts, train_images, test_images)

    statistics_of_dataset(class_counts, train_images, test_images)

    if replace_files:
        move_files(test_images)

    if rollback:
        rollback_files(CONST.dir_train_images, CONST.dir_test_images)
        rollback_files(CONST.dir_train_masks, CONST.dir_test_mask)


if __name__ == "__main__":
    main(replace_files=True, rollback=False)
