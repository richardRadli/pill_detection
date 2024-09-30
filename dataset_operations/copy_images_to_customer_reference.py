import os
import random
import re
import shutil

from tqdm import tqdm

from config.dataset_paths_selector import dataset_images_path_selector
from utils.utils import file_reader


def main(operation="reference"):
    random.seed(42)

    src_imgs = dataset_images_path_selector("ogyei").get("unsplitted").get("images")
    src_labels = dataset_images_path_selector("ogyei").get("unsplitted").get("segmentation_labels")

    images = (
        "customer_images" if operation == "customer"
        else ("reference_images" if operation == "reference"
              else None)
    )

    masks = (
        "customer_mask_images" if operation == "customer"
        else ("reference_mask_images" if operation == "reference"
              else None)
    )

    labels = (
        "customer_segmentation_labels" if operation == "customer"
        else ("reference_segmentation_labels" if operation == "reference"
              else None)
    )

    dst_imgs = dataset_images_path_selector("ogyei").get(operation).get(images)
    dst_masks = dataset_images_path_selector("ogyei").get(operation).get(masks)
    dst_labels = dataset_images_path_selector("ogyei").get(operation).get(labels)

    src_img_files = file_reader(src_imgs, "png")

    pill_files = {}

    for f in src_img_files:
        filename = os.path.basename(f)
        match = re.search(r'id_\d+_(.+?)_\d+\.png', filename)
        if match:
            pill_name = match.group(1)
            if pill_name not in pill_files:
                pill_files[pill_name] = []
            pill_files[pill_name].append(f)
        else:
            print(f"Filename {filename} doesn't match the expected pattern")

    for pill_name, files in pill_files.items():
        random.shuffle(files)

        if operation == "reference":
            collected_files = files[:2]
        else:
            collected_files = files[2:]

        for file in tqdm(collected_files, total=len(collected_files)):
            image_filenames = os.path.basename(file)
            mask_files = file.replace("images", "gt_masks")
            label_filenames = image_filenames.replace(".png", ".txt")

            shutil.copy(file, str(os.path.join(dst_imgs, image_filenames)))
            shutil.copy(mask_files, str(os.path.join(dst_masks, image_filenames)))
            shutil.copy(os.path.join(src_labels, label_filenames), os.path.join(dst_labels, label_filenames))


if __name__ == "__main__":
    operations = ["reference", "customer"]
    for operation in operations:
        main(operation)
