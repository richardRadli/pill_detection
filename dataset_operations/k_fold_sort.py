import os
import shutil
import random

from tqdm import tqdm

from config.const import DATA_PATH, IMAGES_PATH
from utils.utils import create_timestamp, find_latest_file_in_directory


def folds(load: bool = False, num_folds: int = 5) -> dict:
    """
    Generate or load k-folds of class names.

    :param load: If True, load k-folds from a previously generated file. If False, generate new k-folds.
    :param num_folds: Number of folds.
    :return: A dictionary where keys are fold names (fold1, fold2, ..., fold_{num_fodls}) and values are lists of
    class names.
    """

    if not load:
        timestamp = create_timestamp()
        data_list = os.listdir(os.path.join(IMAGES_PATH.get_data_path("train_rgb_stream_ogyei")))

        k_folds = {f"fold{i + 1}": [] for i in range(num_folds)}
        random.shuffle(data_list)

        for i, class_name in enumerate(data_list):
            fold_index = i % num_folds
            k_folds[f"fold{fold_index + 1}"].append(class_name)

        for fold_name, class_names in k_folds.items():
            print(f"{fold_name}: {class_names}")

        path_to_save = os.path.join(DATA_PATH.get_data_path("k_folds"), f"{timestamp}_k_folds.txt")
        with open(path_to_save, "w") as file:
            for fold_name, class_names in k_folds.items():
                file.write(f"{fold_name}: {', '.join(class_names)}\n")
    else:
        k_folds = {}
        latest_txt_file = find_latest_file_in_directory(DATA_PATH.get_data_path("k_folds"), extension="txt")
        print(latest_txt_file)
        with open(latest_txt_file, "r") as file:
            for line in file:
                fold_name, class_names = line.strip().split(":")
                k_folds[fold_name.strip()] = class_names.strip().split(", ")

    print(k_folds)
    return k_folds


def move_images_to_folds(fold_id: str = "fold1", op: str = "train", op2: str = "ref") -> None:
    """
    Move images from specified fold to a new destination.

    :param fold_id: Identifier for the fold (e.g., fold1, fold2, ..., fold5).
    :param op: Operation type (e.g., "train").
    :param op2: Second operation type (e.g., "ref").
    :return: None
    """

    folders_to_copy = folds().get(fold_id)

    source_root = IMAGES_PATH.get_data_path("stream_images_ogyei")
    destination_root = os.path.join(IMAGES_PATH.get_data_path("stream_images_ogyei_test"), op2)

    source_subdirs = ['contour', 'lbp', 'rgb', 'texture']
    destination_subdirs = ['contour', 'lbp', 'rgb', 'texture']

    for source_dir, dest_dir in zip(source_subdirs, destination_subdirs):
        source_dir = os.path.join(source_root, source_dir)
        dest_dir = os.path.join(destination_root, dest_dir)

        for folder in folders_to_copy:
            source_path = os.path.join(source_dir, op, folder)
            dest_path = os.path.join(dest_dir, folder.lower())

            if os.path.exists(source_path):
                os.makedirs(dest_path, exist_ok=True)

                for item in os.listdir(source_path):
                    source_item = os.path.join(source_path, item)
                    dest_item = os.path.join(dest_path, item)
                    shutil.move(source_item, dest_item)
                print(f"Copied {folder} from {source_path} to {dest_path}")
            else:
                print(f"Folder {folder} not found in {source_path}")


def delete_empty_subdirectories(root_path, delete_empty=True) -> None:
    """
    Delete empty subdirectories within the specified root path.

    :param root_path: The root path to search for empty subdirectories.
    :param delete_empty: If True, delete empty directories; otherwise, just print a message.
    :return: None
    """

    for dir_path, dir_names, filenames in os.walk(root_path, topdown=False):
        for dirname in dir_names:
            dir_to_check = os.path.join(dir_path, dirname)
            if os.path.isdir(dir_to_check) and not os.listdir(dir_to_check):
                if delete_empty:
                    print(f"Deleting empty directory: {dir_to_check}")
                    os.rmdir(dir_to_check)
                else:
                    print(f"Found empty directory: {dir_to_check}")


def clean_up_empty_dirs() -> None:
    """
    Clean up empty subdirectories within the specified main and subdirectories.
    :return: None
    """

    main_dirs = ['contour', 'lbp', 'rgb', 'texture']
    sub_dirs = ["train", "valid"]
    for main_dir in main_dirs:
        for sub_dir in sub_dirs:
            root_path = os.path.join(IMAGES_PATH.get_data_path("stream_images_ogyei"), main_dir, sub_dir)
            if os.path.exists(root_path):
                delete_empty_subdirectories(root_path)
                print("Empty subdirectories deleted.")
            else:
                print("Root path does not exist.")


def move_hardest_samples():
    main_dirs = ['contour', 'lbp', 'rgb', 'texture']
    main_dirs_2 = ['contour_hardest', 'lbp_hardest', 'rgb_hardest', 'texture_hardest']
    sub_dirs_train = ["train", "valid"]

    for _, (main_dir, main_dir_2) in tqdm(enumerate(zip(main_dirs, main_dirs_2)), total=len(main_dirs)):
        dest_path = (
                "C:/Users/ricsi/Documents/project/storage/IVM/images/hardest_samples/efficient_net/%s" % main_dir_2)
        for sub_dir_tr in tqdm(sub_dirs_train, total=len(sub_dirs_train)):
            source_path = \
                ("C:/Users/ricsi/Documents/project/storage/IVM/images/stream_images/ogyei/%s/%s" % (main_dir, sub_dir_tr))
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)


def move_images_back_to_train() -> None:
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
            src_path = os.path.join(IMAGES_PATH.get_data_path("stream_images_ogyei_test"), sub_dirs_test, category_dir)
            dst_path = os.path.join(IMAGES_PATH.get_data_path("stream_images_ogyei"), category_dir, sub_dirs_train)

            if os.path.exists(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"Images moved from {src_path} to {dst_path}")
            else:
                print(f"Source path {src_path} does not exist.")


def erase_files():
    """
    Remove directories corresponding to the test set.
    :return: None
    """

    sub_dirs_trains = ["train", "valid"]
    sub_dirs_tests = ["ref", "query"]

    for _, (sub_dirs_train, sub_dirs_test) in tqdm(enumerate(zip(sub_dirs_trains, sub_dirs_tests)),
                                                   total=len(sub_dirs_trains),
                                                   desc="Erasing files"):
        src_path = os.path.join(IMAGES_PATH.get_data_path("stream_images_ogyei_test"), sub_dirs_test)

        if os.path.exists(src_path):
            shutil.rmtree(src_path)
            print(f"Directory erased: {src_path}")
        else:
            print(f"Source path {src_path} does not exist.")


if __name__ == "__main__":
    a = folds(load=True)
    move_images_to_folds("fold1", "train", "ref")
    # move_images_to_folds("fold5", "valid", "query")
    # clean_up_empty_dirs()
    # move_hardest_samples()
    # rollback_folds()
    # erase_files()
