import re
import os
import shutil

from glob import glob
from tqdm import tqdm

from const import CONST
from utils import numerical_sort


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- F I N D   T E S T   F I L E S -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_test_files() -> list:
    """
    This function searches for files in a specified directory that match a certain naming pattern, and moves them to a
    new directory while also returning their names in a list.

    :return: test_file -> list containing the names of the moved files without their extensions.
    """

    dest_dir = CONST.dir_test_images
    # Define the regular expression pattern to match the file names
    pattern = r'^\d+_[^_]+_[^_]+_[^_]+_[^_]+_(.*)\.png$'

    path = CONST.dir_train_images
    file_names = sorted(glob(path + "/*.png"), key=numerical_sort)
    prev_groups = None

    cnt = 0
    test_file = []

    for file_name in file_names:
        file_name = (file_name.split("\\")[2])
        match = re.match(pattern, file_name)
        if match:
            # Extract the groups from the pattern match
            groups = match.groups()
            # Check if there is a change in the groups
            if prev_groups is not None and groups != prev_groups:
                test_file.append(file_name.split(".")[0])

                dest_file = os.path.join(dest_dir, file_name)
                src_file = os.path.join(path, file_name)
                shutil.move(src_file, dest_file)
                cnt += 1

            prev_groups = groups
        else:
            print('No match for:', file_name)

    print(f"{cnt} images were moved")

    return test_file


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------- F I N D   M A S K   T E S T   F I L E S --------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_mask_test_files(test_file: list) -> None:
    """
    Function takes in a list of test file names and searches for corresponding mask files in a directory

    :param test_file:
    :return: None
    """

    path = CONST.dir_train_masks
    dest_dir = CONST.dir_test_mask

    files = sorted(glob(path + "/*.png"), key=numerical_sort)

    cnt = 0

    for file_name in files:
        file_name = file_name.split('\\')[2]
        for test_name in test_file:
            if test_name in file_name:
                dest_file = os.path.join(dest_dir, file_name)
                src_file = os.path.join(path, file_name)
                shutil.move(src_file, dest_file)
                cnt += 1

    print(f"{cnt} mask files were moved")


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ C R E A T E   L A B E L   D I R S -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def create_label_dirs(input_path: str) -> None:
    """
    Function create labels. Goes through a directory, yield the name of the medicine(s) from the file name, and create
    a corresponding directory, if that certain directory does not exist. Finally, it copies every image with the same
    label to the corresponding directory.

    :param input_path: string, path to the directory.
    :return: None
    """

    files = os.listdir(input_path)

    for file in tqdm(files):
        if file.endswith(".png"):
            match = re.search(r"\d{3,4}_(.+)_s", file)
            if match:
                value = match.group(1)
                out_path = (os.path.join(input_path, value))
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                    print(f"Directory {value} has been created!")
                shutil.move(os.path.join(input_path, file), out_path)


def create_cure_dataset():
    root_dir = "D:/project/IVM/images/Pill_Images"
    new_root_dir = "D:/project/IVM/images/Pill_Images_new"

    # create new root directory
    if not os.path.exists(new_root_dir):
        os.mkdir(new_root_dir)

    # loop through label class directories
    for label_dir in os.listdir(root_dir):
        label_dir_path = os.path.join(root_dir, label_dir)

        # loop through bottom and top directories
        for sub_dir in os.listdir(label_dir_path):
            sub_dir_path = os.path.join(label_dir_path, sub_dir)

            # loop through customer and ref directories
            for sub_sub_dir in os.listdir(sub_dir_path):
                sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)

                # copy image from ref directory to new directory
                if sub_sub_dir == "Reference":
                    image_name = os.listdir(sub_sub_dir_path)[0]
                    image_path = os.path.join(sub_sub_dir_path, image_name)

                    new_label_dir_path = os.path.join(new_root_dir, label_dir)

                    if not os.path.exists(new_label_dir_path):
                        os.mkdir(new_label_dir_path)

                    new_image_path = os.path.join(new_label_dir_path, image_name)
                    shutil.copy(image_path, new_image_path)

                # copy images from customer directory to new directory
                elif sub_sub_dir == "Customer":
                    customer_dir_path = os.path.join(sub_dir_path, sub_sub_dir)
                    for idx, image_name in tqdm(enumerate(os.listdir(customer_dir_path)),
                                                total=len(os.listdir(customer_dir_path)),
                                                desc="Copying images"):
                        image_path = os.path.join(customer_dir_path, image_name)
                        new_label_dir_path = os.path.join(new_root_dir, label_dir)
                        if not os.path.exists(new_label_dir_path):
                            os.mkdir(new_label_dir_path)
                        new_image_path = os.path.join(new_label_dir_path, image_name)
                        shutil.copy(image_path, new_image_path)

create_cure_dataset()
