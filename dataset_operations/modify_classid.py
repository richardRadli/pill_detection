import os

from config.config import ConfigAugmentation
from config.config_selector import dataset_images_path_selector


def path_select(cfg):
    train_annotations = dataset_images_path_selector(cfg.dataset_name).get("train_yolo_labels")
    valid_annotations = dataset_images_path_selector(cfg.dataset_name).get("valid_yolo_labels")
    test_annotations = dataset_images_path_selector(cfg.dataset_name).get("test_yolo_labels")

    return train_annotations, valid_annotations, test_annotations


def modify_class_id(file_name, dst_path):
    try:
        filename = os.path.basename(file_name)
        class_id = int(filename.split('_')[0])

        with open(file_name, 'r') as file:
            content = file.read()

        words = content.split()
        modified_content = content.replace(words[0], str(class_id), 1)

        dst_filename = os.path.join(dst_path, os.path.basename(file_name))
        with open(dst_filename, 'w') as file:
            file.write(modified_content)

    except Exception as e:
        print(e)


def main(operation: str = "train"):
    cfg = ConfigAugmentation().parse()
    train_annotations, valid_annotations, test_annotations = path_select(cfg)

    directory_path = train_annotations if operation == "train" else (
        valid_annotations if operation == "valid" else test_annotations
    )

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            modify_class_id(file_path, directory_path)


if __name__ == '__main__':
    main()
