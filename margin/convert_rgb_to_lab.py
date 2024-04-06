import colorspacious
import json
import os

from config.config import ConfigStreamImages
from config.config_selector import dataset_images_path_selector
from utils.utils import NumpyEncoder, create_timestamp, find_latest_file_in_directory, find_latest_directory


def concatenate_json_files(json_path, output_path):
    concatenated_data = {}

    for file_name in os.listdir(json_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_path, file_name)

            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    concatenated_data.setdefault(key, {}).update(value)

    with open(output_path, "w") as json_file:
        json.dump(concatenated_data, json_file, indent=4)

    return concatenated_data


def rgb_to_lab(load: bool = True):
    cfg = ConfigStreamImages().parse()
    timestamp = create_timestamp()

    json_path = dataset_images_path_selector(cfg.dataset_type).get("dynamic_margin").get("colour_vectors")
    json_file_name_lab = os.path.join(json_path, f"{timestamp}_colors_lab.json")

    if load:
        json_file_name_rgb = find_latest_file_in_directory(json_path, "json")
        with open(json_file_name_rgb, "r") as file:
            concatenated_data = json.load(file)
    else:
        json_file_name_rgb = os.path.join(json_path, f"{timestamp}_colors_rgb.json")
        latest_json_path = find_latest_directory(json_path)
        concatenated_data = concatenate_json_files(latest_json_path, json_file_name_rgb)

    lab_values = {}

    for key, value in concatenated_data.items():
        for image_key, rgb_value in value.items():
            rgb_values = list(rgb_value.values())
            lab_values.setdefault(key, []).append(colorspacious.cspace_convert(rgb_values, "sRGB255", "CIELab"))

    with open(json_file_name_lab, "w") as json_file:
        json.dump(lab_values, json_file, cls=NumpyEncoder)


if __name__ == "__main__":
    rgb_to_lab(load=False)
