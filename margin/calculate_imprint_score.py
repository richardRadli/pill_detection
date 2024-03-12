import json
import numpy as np
import os
import openpyxl

from sklearn.preprocessing import OneHotEncoder
from typing import Dict, List

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector
from utils.utils import create_timestamp, sort_dict, NumpyEncoder


def encoding(words: List[str]) -> Dict[str, List[float]]:
    """
    Encode a list of words using one-hot encoding.

    Args:
        words (List[str]): List of words to encode.

    Returns:
        Dict[str, List[float]]: A dictionary where each word is paired with its one-hot encoded vector as a list of
        floats.
    """

    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(np.array(words).reshape(-1, 1))
    return {word: encoded_features[idx].tolist() for idx, word in enumerate(words)}


def create_feature_vectors(sheet, words: list, feature_type: str) -> dict:
    """
    Create feature vectors based on input words and feature type.

    Args:
        sheet (Workbook): Excel sheet object.
        words (list): List of words for encoding.
        feature_type (str): Type of feature vectors to use.

    Returns:
        dict: Dictionary containing feature vectors.
    """

    encoded_dict = encoding(words)
    feature_dict = {}

    for row in sheet.iter_rows(min_row=3, values_only=True):
        pill_id = row[1]
        pill_id_prefix = pill_id.split("_")[0]

        selected_column = row[4] if feature_type == "imprint_vectors" else row[7]
        if selected_column is None:
            selected_column = "None"

        if pill_id_prefix not in feature_dict:
            feature_dict[pill_id_prefix] = {"top": [], "bottom": []}

        encoded_feature = encoded_dict.get(selected_column)

        if "top" in pill_id:
            feature_dict[pill_id_prefix]["top"].append(encoded_feature)
        elif "bottom" in pill_id:
            feature_dict[pill_id_prefix]["bottom"].append(encoded_feature)
        else:
            raise ValueError(f"Wrong pill_id: {pill_id}")

    return feature_dict


def process_vectors(sheet, words: list, dataset_name: str, timestamp: str, feature_type: str) -> None:
    """
    Process feature vectors, sort them, and save them to a JSON file.

    Args:
        sheet (Workbook): Excel sheet object.
        words (list): List of words for encoding.
        dataset_name (str): Name of the dataset.
        timestamp (str): Timestamp for the file name.
        feature_type (str): Type of feature vectors.

    Returns:
        None
    """
    
    dictionary = create_feature_vectors(sheet, words, feature_type)
    sorted_dict = sort_dict(dictionary)
    path = dataset_images_path_selector(dataset_name).get("dynamic_margin").get(f"{feature_type}")
    json_save_filename = os.path.join(path, f"{timestamp}_{feature_type}.json")
    with open(json_save_filename, "w") as json_file:
        json.dump(sorted_dict, json_file, cls=NumpyEncoder)


def main() -> None:
    """
    
    Returns:
         None
    """
    
    cfg = ConfigAugmentation().parse()
    timestamp = create_timestamp()

    imprint_words = ['DEBOSSED', 'None', 'EMBOSSED', 'PRINTED']
    score_words = [1, 2, 4]

    pill_desc_path = dataset_images_path_selector(cfg.dataset_name).get("dynamic_margin").get("pill_desc_xlsx")
    pill_desc_file = os.path.join(pill_desc_path, "pill_desc.xlsx")

    workbook = openpyxl.load_workbook(pill_desc_file)
    sheet = workbook['Sheet1']

    process_vectors(sheet, imprint_words, cfg.dataset_name, timestamp, "imprint_vectors")
    process_vectors(sheet, score_words, cfg.dataset_name, timestamp, "score_vectors")


if __name__ == '__main__':
    main()
