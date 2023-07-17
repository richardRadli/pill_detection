import os
import random
import shutil

from config.const import DATASET_PATH


def split_files(source_dir, train_dir, valid_dir, test_dir, ratio=(0.6, 0.2, 0.2)):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = os.listdir(source_dir)
    random.shuffle(files)

    total_files = len(files)
    train_files = files[:int(total_files * ratio[0])]
    valid_files = files[int(total_files * ratio[0]):int(total_files * (ratio[0] + ratio[1]))]
    test_files = files[int(total_files * (ratio[0] + ratio[1])):]

    for file in train_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.copy(src, dst)

    for file in valid_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(valid_dir, file)
        shutil.copy(src, dst)

    for file in test_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(test_dir, file)
        shutil.copy(src, dst)


if __name__ == "__main__":
    source_dir = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/cure/Customer_bbox/"
    train_dir = DATASET_PATH.get_data_path("cure_train")
    valid_dir = DATASET_PATH.get_data_path("cure_valid")
    test_dir = DATASET_PATH.get_data_path("cure_test")

    # Create the train, valid, and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Split the files by class and copy them to the corresponding directories
    split_files(source_dir, train_dir, valid_dir, test_dir)
