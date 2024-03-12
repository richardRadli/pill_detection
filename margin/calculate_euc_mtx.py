import itertools
import json
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector, Fourier_configs


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
    json_path = dataset_images_path_selector(cfg.dataset_name).get("other").get("pill_colours_rgb_lab")
    json_file_name = os.path.join(json_path, f"colors_{cfg.dataset_name}_lab.json")

    with open(json_file_name, "r") as file:
        lab_values = json.load(file)

    norm_lab_values = normalize_lab_values(lab_values)

    # Fourier desc
    fourier_desc_path = Fourier_configs(cfg.dataset_name).get("Fourier_saved_mean_vectors")
    fourier_desc_file = os.path.join(fourier_desc_path, "2024-03-11_09-52-53_pill_coeffs_order_15.json")

    with open(fourier_desc_file, "r") as file:
        fourier_desc_values = json.load(file)

    asd = (norm_lab_values["0"]["bottom"])
    fd = fourier_desc_values["0"]["bottom"]
    asd = (list(itertools.chain.from_iterable(asd)))
    print(fd + asd)

    words = ["PRINTED", "DEBOSSED", "EMBOSSED"]
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(np.array(words).reshape(-1, 1))
    print(encoded_features)


if __name__ == "__main__":
    main()
