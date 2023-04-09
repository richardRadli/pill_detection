import concurrent.futures
import numpy as np
import os
import re

from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm

from const import CONST


# ------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- L O A D   F I L E S ----------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def load_files():
    if not os.path.isdir(CONST.dir_train_images):
        raise ValueError(f"Invalid path: {CONST.dir_train_images} is not a directory")

    if not os.path.isdir(CONST.dir_labels_data):
        raise ValueError(f"Invalid path: {CONST.dir_labels_data} is not a directory")

    image_files = sorted([str(file) for file in Path(CONST.dir_train_images).glob("*.jpg")] +
                         [str(file) for file in Path(CONST.dir_train_images).glob("*.png")])

    text_files = sorted([str(file) for file in Path(CONST.dir_labels_data).glob("*.txt")])

    if not image_files:
        raise ValueError(f"No image files found in {CONST.dir_train_images}")

    if not text_files:
        raise ValueError(f"No text files found in {CONST.dir_labels_data}")

    return image_files, text_files


# ------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- R E M O V E   R O B O F L O W   S U F F I X ------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def remove_roboflow_suffix(file_name):
    new_name = None

    if file_name.endswith(".jpg"):
        new_name = re.sub("\\.rf\\..+\\.jpg$", "", file_name)
        new_name = new_name.replace("_png", ".png")
        os.rename(file_name, new_name)
    elif file_name.endswith(".txt"):
        new_name = re.sub("\\.rf\\..+\\.txt$", "", file_name)
        new_name = new_name.replace("_png", ".txt")
        os.rename(file_name, new_name)
    else:
        pass

    return new_name


# ------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- P R O C E S S   D A T A --------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def process_data(img_files: str, txt_files: str):
    try:
        img = Image.open(img_files)
        img_width, img_height = img.size
    except FileNotFoundError:
        print(f"Error: {img_files} is not a valid image file.")
        return None, None

    try:
        with open(txt_files, "r") as file:
            line = file.readline().strip()
            yolo_coords = line.split()[1:]
            yolo_coords = [float(x.strip('\'')) for x in yolo_coords]
    except FileNotFoundError:
        print(f"Error: {txt_files} is not a valid text file.")
        return None, None

    coords = [int(coord * img_width if i % 2 == 0 else coord * img_height) for i, coord in enumerate(yolo_coords)]

    mask = Image.new('1', (img_width, img_height), 0)
    xy = list(zip(coords[::2], coords[1::2]))
    ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask)

    return img, mask


# ------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- S A V E   M A S K S ----------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def save_masks(mask: np.ndarray, img_file: str):
    name = img_file.split("\\")[-1]
    save_path = (os.path.join(CONST.dir_test_mask, name))
    mask_pil = Image.fromarray(mask)
    mask_pil.save(save_path)


# ------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------- M A I N ----------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def main(ditch_suffix: bool = True, save: bool = True):
    img_files, txt_files = load_files()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for img_file, txt_file in zip(img_files, txt_files):
            if ditch_suffix:
                img_file = remove_roboflow_suffix(img_file)
                txt_file = remove_roboflow_suffix(txt_file)
            futures.append(executor.submit(process_data, img_file, txt_file))

        for future, (img_file, _) in tqdm(zip(futures, zip(img_files, txt_files)), total=len(img_files),
                                          desc="Processing data"):
            try:
                img, mask = future.result()
                if save:
                    save_masks(mask, img_file)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")


if __name__ == "__main__":
    main(ditch_suffix=False, save=True)
