import os
import re

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def path_select(cfg):
    train_annotations = dataset_images_path_selector(cfg.dataset_name).get("train").get("yolo_labels")
    valid_annotations = dataset_images_path_selector(cfg.dataset_name).get("valid").get("yolo_labels")
    test_annotations = dataset_images_path_selector(cfg.dataset_name).get("test").get("yolo_labels")

    return train_annotations, valid_annotations, test_annotations


def main(operation: str = "test"):
    cfg = ConfigAugmentation().parse()
    train_annotations, valid_annotations, test_annotations = path_select(cfg)

    directory_path = train_annotations if operation == "train" else (
        valid_annotations if operation == "valid" else test_annotations
    )
    class_id_mapping = {}

    for filename in sorted_alphanumeric(os.listdir(directory_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            try:
                filename = os.path.basename(file_path)
                class_id = int(filename.split('_')[0])

                with open(file_path, 'r') as file:
                    content = file.read()

                words = content.split()

                if class_id not in class_id_mapping:
                    class_id_mapping[class_id] = len(class_id_mapping)

                modified_content = content.replace(words[0], str(class_id_mapping[class_id]) + ' ', 1)

                dst_filename = os.path.join(directory_path, os.path.basename(file_path))
                with open(dst_filename, 'w') as file:
                    file.write(modified_content)

            except Exception as e:
                print(e)


if __name__ == '__main__':
    main()
