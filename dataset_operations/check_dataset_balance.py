import logging
import matplotlib.pyplot as plt
import os
import pandas as pd

from typing import Dict

from config.logger_setup import setup_logger
from config.const import DATASET_PATH


# ------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- P A T H   S E L E C T O R -------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
def path_selector(op: str):
    """
    Selects the correct directory paths based on the given operation string.

    :param op: A string indicating the operation mode (train or test).
    :return: A dictionary containing directory paths for images and dataset name.
    :raises ValueError: If the operation string is not "train" or "test".
    """

    if op.lower() == "ogyi":
        path_to_images = {
            "dataset_name": "ogyi",
            "dataset_directory": DATASET_PATH.get_data_path("ogyi_v2_unsplitted_images"),
        }
    elif op.lower() == "cure":
        path_to_images = {
            "dataset_name": "cure",
            "dataset_directory": DATASET_PATH.get_data_path("cure_customer"),
        }
    else:
        raise ValueError("Wrong operation!")

    return path_to_images


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- C A L C U L A T E   P R O P O R T I O N S --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def calculate_proportions(dir_images: str, dataset_name: str) -> [Dict, Dict]:
    """
    Calculate the proportions of the dataset and return a dictionary with class names as keys and their respective
    counts.

    :param dir_images: Directory path of the image files.
    :param dataset_name: name of the dataset, either "ogyi" or "cure".
    :return: A dictionary with class names as keys and their respective counts in the dataset, and another dictionary
    with the total number of class instances.
    """
    class_counts = {}
    total_count = 0

    for filename in os.listdir(dir_images):
        if filename.endswith('.png'):
            class_name = '_'.join(filename.split('_')[2:-1]) if dataset_name == "ogyi" else filename.split('_')[0]
            class_counts.setdefault(class_name, 0)
            class_counts[class_name] += 1
            total_count += 1

    proportions = {class_name: (count / total_count) * 100 for class_name, count in class_counts.items()}

    return proportions, class_counts


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------- C A L C U L A T E   I M B A L A N C E   R A T I O ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def calculate_imbalance_ratio(class_counts: dict) -> None:
    """
    Calculate the imbalance ratio of the dataset.

    :param class_counts: A dictionary with class names as keys and their respective counts in the dataset.
    :return: None
    """
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    logging.info(f'Imbalance ratio of the dataset is: {imbalance_ratio}')


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- P L O T   D A T A --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def plot_data(dataset_name: str, proportions: dict, class_counts: dict, threshold: float = 0.5) -> None:
    """
    Plot the data as a dataframe and a bar plot.

    :param dataset_name:
    :param proportions: A dictionary with class names as keys and their respective proportions in the dataset.
    :param class_counts: A dictionary with class names as keys and their respective counts in the dataset.
    :param threshold: Threshold value for class proportions.
    :return: None
    """

    df = pd.DataFrame.from_dict(proportions, orient='index', columns=['Proportion'])
    df.index.name = 'Class'
    df.sort_values(by=['Proportion'], ascending=False, inplace=True)

    # Add instance counts column to the DataFrame
    df['Instances'] = [class_counts.get(class_name, 0) for class_name in df.index]

    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    if dataset_name == "cure":
        df.index = pd.to_numeric(df.index)
    df = df.sort_index()

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

    plt.bar(df.index, df['Proportion'], color=bar_colors, label='Proportion')
    plt.axhline(y=threshold_value_above, color='darkorange', linestyle='--', linewidth=1, label='Threshold above')
    plt.axhline(y=threshold_value_below, color='peru', linestyle='--', linewidth=1, label='Threshold below')
    plt.xlabel('Class')
    plt.ylabel('Proportion')
    plt.title('Class Proportions')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main(name_of_dataset: str) -> None:
    """
    Executes the functions above.
    :param name_of_dataset: Name of the dataset, could be either "ogyi" or "cure".
    :return: None
    """

    setup_logger()
    dataset_info = path_selector(op=name_of_dataset)
    class_proportions, class_counts = \
        calculate_proportions(dataset_info.get("dataset_directory"), dataset_info.get("dataset_name"))
    calculate_imbalance_ratio(class_proportions)
    plot_data(dataset_info.get("dataset_name"), class_proportions, class_counts)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- __M A I N__ -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main(name_of_dataset="cure")
