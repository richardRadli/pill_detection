"""
File: check_dataset_balance.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: May 03, 2023

Description: This program performs data analysis on a dataset. It calculates the proportions of different classes in the
dataset, plots the data as a dataframe and a bar plot, and calculates the imbalance ratio of the dataset.
"""

import logging
import matplotlib.pyplot as plt
import os
import pandas as pd

from config.logger_setup import setup_logger
from config.const import DATASET_PATH


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ G E T   C L A S S E S -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_classes(dir_images: str) -> dict:
    """
    Get the classes present in the dataset.

    :param dir_images: Directory path of the image files.
    :return: A dictionary with class names as keys and initial count set to 0.
    """

    classes = set()

    for filename in os.listdir(dir_images):
        if filename.endswith('.png'):
            class_name = '_'.join(filename.split('_')[2:-1])
            classes.add(class_name)

    classes = sorted(classes)

    return {class_name: 0 for class_name in classes}


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- C A L C U L A T E   P R O P O R T I O N S --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def calculate_proportions(dir_images: str, class_counts: dict) -> dict:
    """
    This function calculates the proportions of the dataset.
    :param dir_images: Directory path of the image files.
    :param class_counts: A dictionary with class names.
    :return: A dictionary with class names as keys and count set to the according proportion value.
    """

    for filename in os.listdir(dir_images):
        if filename.endswith('.png'):
            class_name = '_'.join(filename.split('_')[2:-1])
            class_counts[class_name] += 1

    total_count = len(os.listdir(dir_images))
    proportions = {}

    for class_name, count in class_counts.items():
        proportion = (count / total_count) * 100
        proportions[class_name] = proportion

    return proportions


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- P L O T   D A T A -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_data(proportions: dict, class_counts: dict, threshold: float = 0.5) -> None:
    """
    Plots the data first az dataframe, displaying the class names, the proportion values and the number of instances for
    all classes. Plots the proportions by classes, green bars indicates the good proportion classes, red indicates the
    ones that are above the threshold, blue indicates the ones that are under the threshold.
    :param proportions: A dictionary with class names as keys and count set to the according proportion value.
    :param class_counts: A dictionary with class names.
    :param threshold: Threshold value.
    :return: None
    """

    df = pd.DataFrame.from_dict(proportions, orient='index', columns=['Proportion'])
    df.index.name = 'Class'
    df['Instances'] = [class_counts[class_name] for class_name in df.index]
    df.sort_values(by=['Proportion'], ascending=False, inplace=True)

    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    logging.info(df)

    # Calculate threshold value
    mean = df['Proportion'].mean()
    std = df['Proportion'].std()
    threshold_value_above = mean + threshold * std
    threshold_value_below = mean - threshold * std

    plt.figure(figsize=(20, 10))
    bar_colors = ['tab:red' if proportion > threshold_value_above
                  else 'tab:blue' if proportion < threshold_value_below
                  else 'tab:green' for proportion in df['Proportion']]

    plt.bar(df.index, df['Proportion'], color=bar_colors)
    plt.axhline(y=threshold_value_above, color='darkorange', linestyle='--', linewidth=1, label='Threshold above')
    plt.axhline(y=threshold_value_below, color='peru', linestyle='--', linewidth=1, label='Threshold below')
    plt.xlabel('Class')
    plt.ylabel('Proportion (%)')
    plt.title('Class Proportions')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- C A L C U L A T E   I M B A L A N C E   R A T I O ---------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def calculate_imbalance_ratio(class_counts: dict) -> None:
    """
    Calculates the imbalance ratio of the dataset.
    :param class_counts: A dictionary with class names.
    :return: None
    """

    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    logging.info(f'Imbalance ratio of the dataset is: {imbalance_ratio}')


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- M A I N ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    setup_logger()
    images_directory = DATASET_PATH.get_data_path("ogyi_v2_unsplitted_images")
    number_of_classes = get_classes(images_directory)
    proportion_value = calculate_proportions(images_directory, number_of_classes)
    calculate_imbalance_ratio(number_of_classes)
    plot_data(proportion_value, number_of_classes)


if __name__ == "__main__":
    main()
