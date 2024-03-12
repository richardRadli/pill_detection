import itertools
import json
import numpy as np
import os

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


def main():
    cfg = ConfigAugmentation().parse()

    # LAB values
    color_vectors_path = dataset_images_path_selector(cfg.dataset_name).get("dynamic_margin").get("colour_vectors")
    color_json_file = find_latest_file_in_directory(color_vectors_path, "json")
    color_json_file_name = os.path.join(color_vectors_path, color_json_file)

    with open(color_json_file_name, "r") as file:
        lab_values = json.load(file)

    norm_lab_values = normalize_lab_values(lab_values)

    # Fourier desc
    fourier_desc_path = (
        dataset_images_path_selector(cfg.dataset_name).get("dynamic_margin").get("Fourier_saved_mean_vectors")

    )
    fourier_json_file = find_latest_file_in_directory(fourier_desc_path, "json")
    fourier_json_file_name = os.path.join(fourier_json_file, color_json_file)

    with open(fourier_json_file_name, "r") as file:
        fourier_desc_values = json.load(file)

    # asd = (norm_lab_values["0"]["bottom"])
    # fd = fourier_desc_values["0"]["bottom"]
    # asd = (list(itertools.chain.from_iterable(asd)))
    # print(fd + asd)


if __name__ == "__main__":
    main()
