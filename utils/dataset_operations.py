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

    path = CONST.dir_img
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

    path = CONST.dir_mask
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

# tf = find_test_files()
# find_mask_test_files(tf)
