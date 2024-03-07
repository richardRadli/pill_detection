import logging
import os
import shutil
import random

from tqdm import tqdm

from config.config import ConfigStreamNetwork
from config.config_selector import dataset_images_path_selector
from utils.utils import create_timestamp, find_latest_file_in_directory


class KFoldSort:
    def __init__(self, load_folds, fold_name, erase):
        self.stream_cfg = ConfigStreamNetwork().parse()
        self.load_folds = load_folds
        self.fold_name = fold_name
        self.erase = erase

    def folds(self, load: bool = False, num_folds: int = None) -> dict:
        """
        Generate or load k-folds of class names.

        :param load: If True, load k-folds from a previously generated file. If False, generate new k-folds.
        :param num_folds: Number of folds.
        :return: A dictionary where keys are fold names (fold1, fold2, ..., fold_{num_folds}) and values are lists of
        class names.
        """

        if not load:
            timestamp = create_timestamp()
            data_list = os.listdir(
                dataset_images_path_selector(
                    self.stream_cfg.dataset_type).get("src_stream_images").get("reference").get("stream_images_rgb")
            )

            k_folds = {f"fold{i + 1}": [] for i in range(num_folds)}
            random.shuffle(data_list)

            for i, class_name in enumerate(data_list):
                fold_index = i % num_folds
                k_folds[f"fold{fold_index + 1}"].append(class_name)

            for fold_name, class_names in k_folds.items():
                logging.info(f"{fold_name}: {class_names}")

            path_to_save = os.path.join(
                dataset_images_path_selector(self.stream_cfg.dataset_type).get("other").get("k_fold"),
                f"{timestamp}_k_folds.txt"
            )
            with open(path_to_save, "w") as file:
                for fold_name, class_names in k_folds.items():
                    file.write(f"{fold_name}: {', '.join(class_names)}\n")
        else:
            k_folds = {}
            latest_txt_file = find_latest_file_in_directory(
                path=dataset_images_path_selector(self.stream_cfg.dataset_type).get("other").get("k_fold"),
                extension="txt"
            )

            logging.info(latest_txt_file)
            with open(latest_txt_file, "r") as file:
                for line in file:
                    fold_name, class_names = line.strip().split(":")
                    k_folds[fold_name.strip()] = class_names.strip().split(", ")

        return k_folds

    def move_images_to_folds(self, sorted_folds, fold_id: str = "fold1", operation: str = "reference",
                             data_role: str = "train") -> None:
        """

        :param sorted_folds:
        :param fold_id:
        :param operation:
        :param data_role:
        :return:
        """

        all_folds = sorted_folds.keys()
        test_classes = sorted_folds.get(fold_id)

        train_classes = []
        for other_fold_id in all_folds:
            if other_fold_id != fold_id:
                train_classes.extend(sorted_folds.get(other_fold_id, []))

        train_classes = sorted(set(train_classes))

        classes_data_role = train_classes if data_role == "train" else \
            (test_classes if data_role == "test" else None)

        if classes_data_role is None:
            raise ValueError(f"{classes_data_role} value is None")

        source_root = (
            dataset_images_path_selector(self.stream_cfg.dataset_type).get("src_stream_images").get(
                operation).get("stream_images")
        )

        if data_role == "train":
            stream_images_dir = "stream_images_anchor" if operation == "customer" else \
            ("stream_images_pos_neg" if operation == "reference" else None)
        elif data_role == "test":
            stream_images_dir = "query" if operation == "customer" else \
            ("ref" if operation == "reference" else None)
        else:
            raise ValueError()

        if stream_images_dir is None:
            raise ValueError(f"{stream_images_dir} value is None")

        destination_root = (
            dataset_images_path_selector(self.stream_cfg.dataset_type).get("dst_stream_images").get(stream_images_dir)
        )

        src_subdirectories = ['contour', 'lbp', 'rgb', 'texture']
        dst_subdirectories = ['contour', 'lbp', 'rgb', 'texture']

        for source_dir, dst_dir in tqdm(zip(src_subdirectories, dst_subdirectories),
                                        total=len(src_subdirectories),
                                        desc="Stream directories"):
            source_dir = os.path.join(source_root, source_dir)
            dst_dir = os.path.join(destination_root, dst_dir)

            for folder in tqdm(classes_data_role,
                               total=len(classes_data_role),
                               desc="Pill folders"):
                source_path = os.path.join(str(source_dir), folder)
                dst_path = os.path.join(str(dst_dir), folder.lower())

                if os.path.exists(source_path):
                    os.makedirs(dst_path, exist_ok=True)

                    for item in (os.listdir(source_path)):
                        source_item = os.path.join(source_path, item)
                        dst_item = os.path.join(dst_path, item)
                        shutil.copy(source_item, dst_item)
                else:
                    logging.error(f"Folder {folder} not found in {source_path}")

    def erase_files(self):
        """
        Remove directories corresponding to the test set.
        :return: None
        """
        root = (
            dataset_images_path_selector(self.stream_cfg.dataset_type).get("dst_stream_images")
        )

        for _, value in root.items():
            logging.info("Removing {}".format(value))
            shutil.rmtree(value)

    def main(self):
        if self.erase:
            self.erase_files()
        if self.load_folds:
            sorted_folds = self.folds(load=True)
        else:
            sorted_folds = self.folds(load=False, num_folds=5)
        self.move_images_to_folds(sorted_folds, self.fold_name, operation="reference", data_role="train")
        self.move_images_to_folds(sorted_folds, self.fold_name, operation="reference", data_role="test")
        self.move_images_to_folds(sorted_folds, self.fold_name, operation="customer", data_role="train")
        self.move_images_to_folds(sorted_folds, self.fold_name, operation="customer", data_role="test")


if __name__ == "__main__":
    k_fold_sort = KFoldSort(load_folds=True, fold_name="fold1", erase=True)
    k_fold_sort.main()
