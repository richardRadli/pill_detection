import os
import random
import shutil


def split_dataset(input_directory, output_val_directory, split_ratio=0.25):
    # Create output validation directory if it doesn't exist
    os.makedirs(output_val_directory, exist_ok=True)

    # Get a list of all subdirectories (class folders) in the input directory
    class_folders = [f for f in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, f))]

    for class_folder in class_folders:
        # Get the list of images in the current class folder
        class_images = [f for f in os.listdir(os.path.join(input_directory, class_folder)) if f.endswith('.png')]
        num_images = len(class_images)

        # Shuffle the list of images
        random.shuffle(class_images)

        # Calculate the split point based on the split ratio
        split_point = int(num_images * split_ratio)

        # Split the images into validation set
        val_images = class_images[:split_point]

        # Move the validation images to the corresponding output validation directory
        for img_file in val_images:
            src_path = os.path.join(input_directory, class_folder, img_file)
            dst_path = os.path.join(output_val_directory, class_folder, img_file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.move(src_path, dst_path)


# Example usage:
input_dir = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/cure/train"
output_val_dir = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/cure/valid"

split_dataset(input_dir, output_val_dir, split_ratio=0.25)
