import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.metrics.pairwise import pairwise_distances

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector
from utils.utils import find_latest_file_in_directory


def normalize_lab_values(lab_values):
    normalized_values = {}

    for key, value in lab_values.items():
        if 'bottom' in value:
            bottom_data = np.array(value['bottom'])
            normalized_bottom_data = (bottom_data - np.min(bottom_data)) / (np.max(bottom_data) - np.min(bottom_data))
        else:
            normalized_bottom_data = None

        if 'top' in value:
            top_data = np.array(value['top'])
            normalized_top_data = (top_data - np.min(top_data)) / (np.max(top_data) - np.min(top_data))
        else:
            normalized_top_data = None

        normalized_values[key] = {
            'bottom': normalized_bottom_data.tolist() if normalized_bottom_data is not None else None,
            'top': normalized_top_data.tolist() if normalized_top_data is not None else None
        }

    return normalized_values


def load_json_files(dataset_name, feature_type: str):
    path = (
        dataset_images_path_selector(dataset_name).get("dynamic_margin").get(feature_type)
    )
    json_file = find_latest_file_in_directory(path, "json")
    json_file_name = os.path.join(path, json_file)

    with open(json_file_name, "r") as file:
        values = json.load(file)

    return values


def flatten_list(input_list):
    concatenated_list = []

    for element in input_list:
        if isinstance(element, list):
            concatenated_list.extend(element)
        else:
            concatenated_list.append(element)

    return concatenated_list


def process_vectors(lab_values, fourier_desc_values, imprint_values, score_values):
    combined_vectors = {}

    for class_id in lab_values.keys():
        combined_vectors[class_id + "_top"] = []
        combined_vectors[class_id + "_bottom"] = []

        combined_sublist_top = (
                lab_values[class_id]['top'] +
                fourier_desc_values[class_id]['top'] +
                imprint_values[class_id]['top'] +
                score_values[class_id]['top']
        )

        combined_sublist_bottom = (
                lab_values[class_id]['bottom'] +
                fourier_desc_values[class_id]['bottom'] +
                imprint_values[class_id]['bottom'] +
                score_values[class_id]['bottom']
        )

        combined_sublist_top = flatten_list(combined_sublist_top)
        combined_sublist_bottom = flatten_list(combined_sublist_bottom)

        combined_vectors[class_id + "_top"].append(combined_sublist_top)
        combined_vectors[class_id + "_bottom"].append(combined_sublist_bottom)

    return combined_vectors


def main():
    cfg = ConfigAugmentation().parse()
    dataset_name = cfg.dataset_name

    # L*a*b* values
    lab_values = load_json_files(dataset_name, "colour_vectors")
    norm_lab_values = normalize_lab_values(lab_values)

    # Fourier descriptors
    fourier_desc_values = load_json_files(dataset_name, "Fourier_saved_mean_vectors")
    norm_fourier_desc_values = normalize_lab_values(fourier_desc_values)

    # imprint vectors
    imprint_values = load_json_files(dataset_name, "imprint_vectors")

    # score vectors
    score_values = load_json_files(dataset_name, "score_vectors")

    combined_vectors = process_vectors(norm_lab_values, norm_fourier_desc_values, imprint_values, score_values)

    labels_list = []
    for key in combined_vectors.keys():
        labels_list.append(key)

    vectors = np.concatenate([np.array(combined_vectors[key]) for key in combined_vectors.keys()])
    distances = pairwise_distances(vectors, metric="euclidean")

    df = pd.DataFrame(distances, index=labels_list, columns=labels_list)

    plt.figure(figsize=(200, 200))
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f", annot_kws={"size": 8}, square=True)

    plt.savefig("C:/Users/ricsi/Desktop/plot.png")


if __name__ == "__main__":
    main()
