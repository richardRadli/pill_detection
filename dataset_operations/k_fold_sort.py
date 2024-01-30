import logging
import os
import shutil
import random

from tqdm import tqdm

from config.config import ConfigStreamNetwork
from config.config_selector import sub_stream_network_configs, dataset_images_path_selector
from utils.utils import create_timestamp, find_latest_file_in_directory


class KFoldSort:
    def __init__(self, load_folds):
        self.stream_cfg = ConfigStreamNetwork().parse()
        self.load_folds = load_folds

    def folds(self, load: bool = False, num_folds: int = None) -> dict:
        """
        Generate or load k-folds of class names.

        :param load: If True, load k-folds from a previously generated file. If False, generate new k-folds.
        :param num_folds: Number of folds.
        :return: A dictionary where keys are fold names (fold1, fold2, ..., fold_{num_fodls}) and values are lists of
        class names.
        """

        if not load:
            timestamp = create_timestamp()
            data_list = (
                os.listdir(
                    sub_stream_network_configs(
                        self.stream_cfg).get(
                        self.stream_cfg.type_of_stream).get(
                        "train").get(
                        self.stream_cfg.dataset_type
                    )
                )
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

    def move_images_to_folds(self, sorted_folds, fold_id: str = "fold1", op: str = "train", op2: str = "ref") -> None:
        """
        Move images from specified fold to a new destination.

        :param sorted_folds:
        :param fold_id: Identifier for the fold (e.g., fold1, fold2, ..., fold5).
        :param op: Operation type (e.g., "train").
        :param op2: Second operation type (e.g., "ref").
        :return: None
        """

        folders_to_copy = sorted_folds.get(fold_id)

        source_root = (
            dataset_images_path_selector(self.stream_cfg.dataset_type).get("other").get("stream_images")
        )
        destination_root = (
            os.path.join(
                dataset_images_path_selector(self.stream_cfg.dataset_type).get("other").get("stream_test_images"),
                op2
            )
        )

        src_subdirectories = ['contour', 'lbp', 'rgb', 'texture']
        dst_subdirectories = ['contour', 'lbp', 'rgb', 'texture']

        for source_dir, dst_dir in zip(src_subdirectories, dst_subdirectories):
            source_dir = os.path.join(source_root, source_dir)
            dst_dir = os.path.join(destination_root, dst_dir)

            for folder in folders_to_copy:
                source_path = os.path.join(str(source_dir), op, folder)
                dst_path = os.path.join(str(dst_dir), folder.lower())

                if os.path.exists(source_path):
                    os.makedirs(dst_path, exist_ok=True)

                    for item in os.listdir(source_path):
                        source_item = os.path.join(source_path, item)
                        dst_item = os.path.join(dst_path, item)
                        shutil.move(source_item, dst_item)
                    print(f"Copied {folder} from {source_path} to {dst_path}")
                else:
                    print(f"Folder {folder} not found in {source_path}")

    @staticmethod
    def delete_empty_subdirectories(root_path, delete_empty=True) -> None:
        """
        Delete empty subdirectories within the specified root path.

        :param root_path: The root path to search for empty subdirectories.
        :param delete_empty: If True, delete empty directories; otherwise, just print a message.
        :return: None
        """

        for dir_path, dir_names, filenames in os.walk(root_path, topdown=False):
            for dir_name in dir_names:
                dir_to_check = os.path.join(dir_path, dir_name)
                if os.path.isdir(dir_to_check) and not os.listdir(str(dir_to_check)):
                    if delete_empty:
                        print(f"Deleting empty directory: {dir_to_check}")
                        os.rmdir(dir_to_check)
                    else:
                        print(f"Found empty directory: {dir_to_check}")

    def clean_up_empty_dirs(self) -> None:
        """
        Clean up empty subdirectories within the specified main and subdirectories.
        :return: None
        """

        main_dirs = ['contour', 'lbp', 'rgb', 'texture']
        sub_dirs = ["train", "valid"]

        for main_dir in main_dirs:
            for sub_dir in sub_dirs:
                root_path = (
                    os.path.join(
                        dataset_images_path_selector(self.stream_cfg.dataset_type).get("other").get("stream_images"),
                        main_dir,
                        sub_dir
                    )
                )
                if os.path.exists(root_path):
                    self.delete_empty_subdirectories(root_path)
                    print("Empty subdirectories deleted.")
                else:
                    print("Root path does not exist.")

    def rollback_folds(self) -> None:
        """
        Move images from the test set back to the training set.
        :return: None
        """

        category_dirs = ['contour', 'lbp', 'rgb', 'texture']
        sub_dirs_trains = ["train", "valid"]
        sub_dirs_tests = ["ref", "query"]

        for _, (sub_dirs_train, sub_dirs_test) in tqdm(enumerate(zip(sub_dirs_trains, sub_dirs_tests)),
                                                       total=len(sub_dirs_trains),
                                                       desc="Rolling back folds"):
            for category_dir in category_dirs:
                src_path = os.path.join(
                    dataset_images_path_selector(self.stream_cfg.dataset_type).get("other").get("stream_test_images"),
                    sub_dirs_test,
                    category_dir
                )
                dst_path = os.path.join(
                    dataset_images_path_selector(self.stream_cfg.dataset_type).get("other").get("stream_test_images"),
                    category_dir,
                    sub_dirs_train
                )

                if os.path.exists(src_path):
                    shutil.copytree(str(src_path), str(dst_path), dirs_exist_ok=True)
                    print(f"Images moved from {src_path} to {dst_path}")
                else:
                    print(f"Source path {src_path} does not exist.")

    def erase_files(self):
        """
        Remove directories corresponding to the test set.
        :return: None
        """

        sub_dirs_trains = ["train", "valid"]
        sub_dirs_tests = ["ref", "query"]

        for _, (sub_dirs_train, sub_dirs_test) in tqdm(enumerate(zip(sub_dirs_trains, sub_dirs_tests)),
                                                       total=len(sub_dirs_trains),
                                                       desc="Erasing files"):
            src_path = os.path.join(
                dataset_images_path_selector(self.stream_cfg.dataset_type).get("other").get("stream_test_images"),
                sub_dirs_test
            )

            if os.path.exists(src_path):
                shutil.rmtree(src_path)
                print(f"Directory erased: {src_path}")
            else:
                print(f"Source path {src_path} does not exist.")

    def main(self):
        if self.load_folds:
            sorted_folds = self.folds(load=True)
        else:
            sorted_folds = self.folds(load=False, num_folds=5)
        # self.move_images_to_folds(sorted_folds, "fold1", "train", "ref")
        # self.move_images_to_folds(sorted_folds, "fold1", "valid", "ref")
        # self.clean_up_empty_dirs()
        # self.rollback_folds()
        # erase_files()


if __name__ == "__main__":
    k_fold_sort = KFoldSort(load_folds=True)
    k_fold_sort.main()
