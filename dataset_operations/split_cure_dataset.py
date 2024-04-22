import os
import random
import shutil

from tqdm import tqdm


def split_images(directory_bbox, train_directory, valid_directory, test_directory, sr):
    # Create train, validation, and test directories if they don't exist
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(valid_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(directory_bbox) if f.endswith('.jpg')]

    # Create a dictionary to store image files by class labels
    class_files = {}
    for file in image_files:
        class_label = file.split('_')[0]
        if class_label not in class_files:
            class_files[class_label] = []
        class_files[class_label].append(file)

    # Shuffle the image files within each class randomly
    for class_label in class_files:
        random.shuffle(class_files[class_label])

    # Calculate the split index for each class based on the split ratio
    train_files = []
    valid_files = []
    test_files = []
    for class_label in class_files:
        split_index1 = int(len(class_files[class_label]) * sr[0])
        split_index2 = int(len(class_files[class_label]) * (sr[0] + sr[1]))
        train_files.extend(class_files[class_label][:split_index1])
        valid_files.extend(class_files[class_label][split_index1:split_index2])
        test_files.extend(class_files[class_label][split_index2:])

    # Move the train files to the train directory
    for file in tqdm(train_files, total=len(train_files), desc="Train images"):
        src = os.path.join(directory_bbox, file)
        dst = os.path.join(train_directory, file)
        shutil.copy(str(src), str(dst))

    # Move the validation files to the validation directory
    for file in tqdm(valid_files, total=len(valid_files), desc="Valid images"):
        src = os.path.join(directory_bbox, file)
        dst = os.path.join(valid_directory, file)
        shutil.copy(str(src), str(dst))

    # Move the test files to the test directory
    for file in tqdm(test_files, total=len(test_files), desc="Test images"):
        src = os.path.join(directory_bbox, file)
        dst = os.path.join(test_directory, file)
        shutil.copy(str(src), str(dst))


if __name__ == '__main__':
    # Split ratio (64% train, 16% validation, 20% test)
    split_ratio = [0.64, 0.16, 0.2]

    directory = "D:/storage/pill_detection/datasets/cure_two_sided/Customer/images_original"
    train_dir = "D:/storage/pill_detection/datasets/cure_two_sided/Customer/train_dir"
    valid_dir = "D:/storage/pill_detection/datasets/cure_two_sided/Customer/valid_dir"
    test_dir = "D:/storage/pill_detection/datasets/cure_two_sided/Customer/test_dir"

    # Call the split_images function
    split_images(directory, train_dir, valid_dir, test_dir, split_ratio)
