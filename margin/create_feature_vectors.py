import json
import logging
import numpy as np
import os

from config.json_config import json_config_selector
from config.dataset_paths_selector import dataset_images_path_selector
from utils.utils import (find_latest_file_in_directory, NumpyEncoder, create_timestamp, plot_euclidean_distances,
                         load_config_json)


def normalize_values(values: dict, operation: str, dataset_name: str = None) -> dict:
    """
    Normalize feature vectors based on the specified operation.

    Args:
        values (dict): Dictionary of feature vectors.
        operation (str): Type of normalization ('lab' or 'fourier').
        dataset_name (str, optional): Dataset name for specific handling. Defaults to None.

    Returns:
        dict: Dictionary of normalized feature vectors.
    """

    normalized_values = {}

    for key, value in values.items():

        data1 = np.array(value[0])
        normalized_data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))

        if operation == 'lab':
            if dataset_name != "cure_one_sided":
                data2 = np.array(value[1])
                normalized_data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
                concat_data = np.concatenate((normalized_data1, normalized_data2), axis=1)
            else:
                concat_data = normalized_data1
        elif operation == "fourier":
            concat_data = normalized_data1
        else:
            raise ValueError("Unknown operation {}".format(operation))

        concat_data_flattened = np.reshape(concat_data, (1, -1))
        normalized_values[key] = concat_data_flattened

    return normalized_values


def load_json_files(dataset_name: str, feature_type: str) -> dict:
    """
    Load feature vectors from a JSON file.

    Args:
        dataset_name (str): Name of the dataset.
        feature_type (str): Type of feature (e.g., 'colour_vectors').

    Returns:
        dict: Dictionary of feature vectors loaded from the JSON file.
    """

    path = (
        dataset_images_path_selector(dataset_name).get("dynamic_margin").get(feature_type)
    )
    json_file = find_latest_file_in_directory(path, "json")
    json_file_name = os.path.join(path, json_file)

    with open(json_file_name, "r") as file:
        values = json.load(file)

    logging.info(f"Loaded json file: {json_file_name}")

    return values


def save_json_file(dataset_name: str, timestamp: str, combined_vectors: dict) -> None:
    """
    Save combined feature vectors to a JSON file.

    Args:
        dataset_name (str): Name of the dataset.
        timestamp (str): Timestamp for file naming.
        combined_vectors (dict): Dictionary of combined feature vectors.

    Returns:
        None
    """

    combined_vectors_path = (
        dataset_images_path_selector(dataset_name).get("dynamic_margin").get("concatenated_vectors")
    )
    combined_vectors_name = os.path.join(combined_vectors_path, f"{timestamp}_concatenated_vectors.json")
    with open(combined_vectors_name, "w") as file:
        json.dump(combined_vectors, file, cls=NumpyEncoder)

    logging.info(f"Saved json file: {combined_vectors_name}")


def reshape_one_hot_encoded_vectors(dataset_name: str, vector_type: str) -> dict:
    """
    Reshape one-hot encoded vectors into a uniform format.

    Args:
        dataset_name (str): Name of the dataset.
        vector_type (str): Type of vector (e.g., 'imprint_vectors').

    Returns:
        dict: Dictionary of reshaped vectors.
    """

    encoded_vectors = load_json_files(dataset_name, vector_type)

    normalized_values = {}

    for key, value in encoded_vectors.items():
        concat_data_flattened = np.reshape(value, (1, -1))
        normalized_values[key] = concat_data_flattened

    return normalized_values


def process_vectors(
    lab_values: dict,
    fourier_desc_values: dict,
    imprint_values: dict,
    score_values: dict
) -> dict:
    """
    Combine multiple feature vectors into a single vector for each class.

    Args:
        lab_values (dict): L*a*b* feature vectors.
        fourier_desc_values (dict): Fourier descriptor vectors.
        imprint_values (dict): Imprint feature vectors.
        score_values (dict): Score feature vectors.

    Returns:
        dict: Dictionary of combined feature vectors.
    """

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
    cfg = (
        load_config_json(
            json_schema_filename=json_config_selector("stream_images").get("schema"),
            json_filename=json_config_selector("stream_images").get("config")
        )
    )
    timestamp = create_timestamp()
    dataset_name = cfg.dataset_type

    # L*a*b* values
    lab_values = load_json_files(dataset_name, "colour_vectors")
    norm_lab_values = normalize_values(lab_values, "lab", dataset_name)

    # Fourier descriptors
    fourier_desc_values = load_json_files(dataset_name, "Fourier_saved_mean_vectors")
    norm_fourier_desc_values = normalize_values(fourier_desc_values, "fourier")

    # imprint vectors
    imprint_values = reshape_one_hot_encoded_vectors(dataset_name, "imprint_vectors")

    # score vectors
    score_values = reshape_one_hot_encoded_vectors(dataset_name, "score_vectors")

    # Combine vectors
    combined_vectors = process_vectors(norm_lab_values, norm_fourier_desc_values, imprint_values, score_values)

    # Save it to a json file
    save_json_file(dataset_name, timestamp, combined_vectors)

    # Plot Euclidean matrix
    plot_euc_dir = dataset_images_path_selector(dataset_name).get("dynamic_margin").get("combined_vectors_euc_dst")
    filename = os.path.join(plot_euc_dir, f"euclidean_distances_{timestamp}.png")
    plot_euclidean_distances(
        vectors=combined_vectors,
        dataset_name=dataset_name,
        filename=filename,
        normalize=False,
        operation="combined_vectors",
        plot_size=80
    )


if __name__ == "__main__":
    main()
