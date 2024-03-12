import colorspacious
import json
import os

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector
from utils.utils import NumpyEncoder, sort_dict, create_timestamp, find_latest_file_in_directory


def rgb_to_lab():
    cfg = ConfigAugmentation().parse()
    timestamp = create_timestamp()

    json_path = dataset_images_path_selector(cfg.dataset_name).get("dynamic_margin").get("colour_vectors")
    json_file_name_rgb = find_latest_file_in_directory(json_path, "json")
    json_file_name_lab = os.path.join(json_path, f"{timestamp}_colors_lab.json")

    with open(json_file_name_rgb, "r") as file:
        json_file = json.load(file)

    rgb_values = {}

    for key, value in json_file.items():
        for rgb_value in value:
            lab_value = colorspacious.cspace_convert(rgb_value, "sRGB255", "CIELab")
            if key in rgb_values:
                rgb_values[key].append(lab_value)
            else:
                rgb_values[key] = [lab_value]

    sorted_dict = sort_dict(rgb_values)

    with open(json_file_name_lab, "w") as json_file:
        json.dump(sorted_dict, json_file, cls=NumpyEncoder)


if __name__ == "__main__":
    rgb_to_lab()
