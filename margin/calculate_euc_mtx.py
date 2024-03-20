import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector
from utils.utils import find_latest_file_in_directory, NumpyEncoder, create_timestamp


def normalize_lab_values(lab_values, type):
    normalized_values = {}

    for key, value in lab_values.items():
        data1 = np.array(value[0])
        normalized_data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))

        data2 = np.array(value[1])
        normalized_data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))

        if type == 'lab':
            concat_data = np.concatenate((normalized_data1, normalized_data2), axis=1)
        else:
            concat_data = np.concatenate((normalized_data1, normalized_data2), axis=0)
        concat_data_flattened = np.reshape(concat_data, (1, -1))
        normalized_values[key] = concat_data_flattened

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


def process_vectors(lab_values, fourier_desc_values, imprint_values, score_values):
    combined_vectors = {}

    for class_id in lab_values.keys():
        combined_vectors[class_id] = []

        combined_vector = np.hstack((
                lab_values[class_id],
                fourier_desc_values[class_id],
                imprint_values[class_id],
                score_values[class_id]
        ))

        combined_vectors[class_id].append(combined_vector)

    combined_vectors = {class_id: values[0] for class_id, values in combined_vectors.items()}
    return combined_vectors


def main():
    cfg = ConfigAugmentation().parse()
    timestamp = create_timestamp()
    dataset_name = cfg.dataset_name

    # L*a*b* values
    lab_values = load_json_files(dataset_name, "colour_vectors")
    norm_lab_values = normalize_lab_values(lab_values, "lab")

    # Fourier descriptors
    fourier_desc_values = load_json_files(dataset_name, "Fourier_saved_mean_vectors")
    norm_fourier_desc_values = normalize_lab_values(fourier_desc_values, "fourier")

    # imprint vectors
    imprint_values = load_json_files(dataset_name, "imprint_vectors")

    # score vectors
    score_values = load_json_files(dataset_name, "score_vectors")

    combined_vectors = process_vectors(norm_lab_values, norm_fourier_desc_values, imprint_values, score_values)

    combined_vectors_path = (
        dataset_images_path_selector(dataset_name).get("dynamic_margin").get("concatenated_vectors")
    )
    combined_vectors_name = os.path.join(combined_vectors_path, f"{timestamp}_concatenated_vectors.json")
    with open(combined_vectors_name, "w") as file:
        json.dump(combined_vectors, file, cls=NumpyEncoder)

    labels_list = []
    for key in combined_vectors.keys():
        labels_list.append(key)

    vectors = np.concatenate([np.array(combined_vectors[key]) for key in combined_vectors.keys()])
    tsne_vectors = TSNE(n_components=3, learning_rate="auto", init='random', perplexity=3).fit_transform(vectors)
    distances = pairwise_distances(tsne_vectors, metric="euclidean")

    df = pd.DataFrame(distances, index=labels_list, columns=labels_list)

    plt.figure(figsize=(200, 200))
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f", annot_kws={"size": 8}, square=True)

    plt.savefig("C:/Users/ricsi/Desktop/plot_tsne.png")


if __name__ == "__main__":
    main()
