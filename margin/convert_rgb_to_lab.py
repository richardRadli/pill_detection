import colorspacious
import json
import os

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector
from utils.utils import NumpyEncoder


def rgb_to_lab():
    cfg = ConfigAugmentation().parse()

    json_path = dataset_images_path_selector(cfg.dataset_name).get("other").get("pill_colours_rgb_lab")
    json_file_name = os.path.join(json_path, f"colors_{cfg.dataset_name}_rgb.json")
    json_file_name_lab = os.path.join(json_path, f"colors_{cfg.dataset_name}_lab.json")

    with open(json_file_name, "r") as file:
        json_file = json.load(file)

    rgb_values = {}

    for key, value in json_file.items():
        for rgb_value in value:
            lab_value = colorspacious.cspace_convert(rgb_value, "sRGB255", "CIELab")
            if key in rgb_values:
                rgb_values[key].append(lab_value)
            else:
                rgb_values[key] = [lab_value]

    if all(key.isdigit() for key in rgb_values.keys()):
        sorted_dict = dict(sorted(rgb_values.items(), key=lambda item: int(item[0])))
    else:
        sorted_dict = rgb_values

    with open(json_file_name_lab, "w") as json_file:
        json.dump(sorted_dict, json_file, cls=NumpyEncoder)


if __name__ == "__main__":
    rgb_to_lab()
