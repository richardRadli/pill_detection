import os
import shutil

from random import sample
from tqdm import tqdm


def copy_files(dir_type):
    # Copy the image file to the train/images directory
    image_source_file = os.path.join(source_dir, "images", file)
    image_target_file = os.path.join(dir_type, "images", file)
    shutil.copyfile(image_source_file, image_target_file)

    # Copy the corresponding text file to the train/labels directory
    text_file = os.path.splitext(file)[0] + ".txt"
    text_source_file = os.path.join(source_dir, "labels", text_file)
    text_target_file = os.path.join(dir_type, "labels", text_file)
    shutil.copyfile(text_source_file, text_target_file)


source_dir = r"C:/Users/ricsi/Documents/project/storage/IVM/datasets/ogyi/full_img_size/unsplitted"
target_dir = r"C:/Users/ricsi/Documents/project/storage/IVM/datasets/ogyi/full_img_size/splitted"

# Create train, valid, and test directories
train_dir = os.path.join(target_dir, "train")
valid_dir = os.path.join(target_dir, "valid")
test_dir = os.path.join(target_dir, "test")

folders = ["labels", "images"]
for f in folders:
    os.makedirs(os.path.join(train_dir, f), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, f), exist_ok=True)
    os.makedirs(os.path.join(test_dir, f), exist_ok=True)

# Get the list of image files
image_files = os.listdir(os.path.join(source_dir, "images"))

# Shuffle the image files
sampled_files = sample(image_files, len(image_files))

# Split the files into train, valid, and test sets
train_image_files = sampled_files[:int(0.7 * len(image_files))]
valid_image_files = sampled_files[int(0.7 * len(image_files)):int(0.85 * len(image_files))]
test_image_files = sampled_files[int(0.85 * len(image_files)):]

# Copy the image and text files to the respective directories
for file in tqdm(train_image_files, total=len(train_image_files), desc="Train images"):
    copy_files(train_dir)

for file in tqdm(valid_image_files, total=len(valid_image_files), desc="Valid images"):
    copy_files(valid_dir)

for file in tqdm(test_image_files, total=len(test_image_files), desc="Testing images"):
    copy_files(test_dir)
