import os
import shutil

from glob import glob
from tqdm import tqdm

from const import CONST
from utils.utils import find_stream_folders


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- G E T   H A R D E S T   S A M P L E S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_hardest_samples(samples_dir: str, sub_dir: str) -> str:
    """
    Returns the path of the .txt file with the latest created timestamp in a subdirectory of the samples' directory.

    :param samples_dir: The path to the samples' directory.
    :param sub_dir: The name of the subdirectory to search for the .txt file.
    :return: The path of the .txt file with the latest created timestamp in the subdirectory.
    """

    directory = os.path.join(samples_dir, sub_dir)
    return max(glob(os.path.join(directory, "*.txt")), key=os.path.getctime)


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ P R O C E S S   T X T -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def process_txt(txt_file: str) -> set:
    """
    Reads a .txt file and extracts a set of paths from its contents.

    :param txt_file: The path to the .txt file.
    :return: A set of paths extracted from the .txt file.
    """

    paths = []

    with open(txt_file, 'r') as f:
        data = eval(f.read())

    for key in data:
        paths.append(key)

    return set(paths)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- F I L E S   T O   M O V E ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def files_to_move(hardest_sample_images: set, src_dir: str) -> list:
    """
    Returns a list of paths to files in a source directory that have a matching name to those in a set of hardest sample
     images.

    :param hardest_sample_images: A set of hardest sample image names to match against.
    :param src_dir: The path to the source directory.
    :return: A list of paths to files in the source directory that match the hardest sample image names.
    """

    list_of_files_to_move = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            copy_of_file = file
            copy_of_file = copy_of_file.replace("contour_", "").replace("texture_", "")
            if copy_of_file in hardest_sample_images:
                list_of_files_to_move.append(os.path.join(root, file))

    return list_of_files_to_move


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- C O P Y   H A R D E S T   S A M P L E S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def copy_hardest_samples(new_dir: str, src_dir: str, hardest_sample_images: list) -> None:
    """
    Copies the hardest sample images from the source directory to a new directory.

    :param new_dir: The path to the new directory.
    :param src_dir: The path to the source directory.
    :param hardest_sample_images: A list of hardest sample image paths to copy.
    :return: None
    """

    for src_paths in tqdm(hardest_sample_images):
        source_path = os.path.join(src_dir, src_paths.split("\\")[2])

        dest_path = src_paths.split("\\")[2]
        dest_path = os.path.join(new_dir, dest_path)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        src_file = os.path.join(source_path, os.path.basename(src_paths))
        dst_file = os.path.join(dest_path, os.path.basename(src_paths))
        shutil.copy(src_file, dst_file)


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- G E T   L A T E S T   T X T   F I L E S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_latest_txt_files(operation: str):
    """
    Retrieves the latest .txt files for contour, RGB, and texture data in the specified directories
    for either positive or negative operations.

    :param operation: The type of operation to perform, either "positive" or "negative".
    :return: A tuple of the paths to the latest contour, RGB, and texture .txt files, respectively.
    :raises ValueError: If the specified operation is neither "positive" nor "negative".
    """

    if operation not in ["positive", "negative"]:
        raise ValueError("Wrong operation!")

    path = CONST.dir_hardest_neg_samples if operation == "negative" else CONST.dir_hardest_pos_samples
    d = find_stream_folders(path)

    latest_contour_txt = get_hardest_samples(path, os.path.join(d[1].split("\\")[-1], "Contour"))
    latest_rgb_txt = get_hardest_samples(path, os.path.join(d[0].split("\\")[-1], "RGB"))
    latest_texture_txt = get_hardest_samples(path, os.path.join(d[2].split("\\")[-1], "Texture"))

    return latest_contour_txt, latest_rgb_txt, latest_texture_txt


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    """

    :return:
    """

    latest_neg_contour_txt, latest_neg_rgb_txt, latest_neg_texture_txt = get_latest_txt_files("negative")
    latest_pos_contour_txt, latest_pos_rgb_txt, latest_pos_texture_txt = get_latest_txt_files("positive")

    hardest_neg_samples_contour = process_txt(latest_neg_contour_txt)
    hardest_neg_samples_rgb = process_txt(latest_neg_rgb_txt)
    hardest_neg_samples_texture = process_txt(latest_neg_texture_txt)

    hardest_pos_samples_contour = process_txt(latest_pos_contour_txt)
    hardest_pos_samples_rgb = process_txt(latest_pos_rgb_txt)
    hardest_pos_samples_texture = process_txt(latest_pos_texture_txt)

    hardest_neg_samples_union = hardest_neg_samples_contour | hardest_neg_samples_rgb | hardest_neg_samples_texture
    hardest_pos_samples_union = hardest_pos_samples_contour | hardest_pos_samples_rgb | hardest_pos_samples_texture
    hardest_samples_union = hardest_pos_samples_union | hardest_neg_samples_union

    result = {x.split('\\')[-1] for x in hardest_samples_union}

    files_to_move_contour = files_to_move(result, CONST.dir_contour)
    copy_hardest_samples(CONST.dir_contour_hardest, CONST.dir_contour, files_to_move_contour)

    files_to_move_rgb = files_to_move(result, CONST.dir_rgb)
    copy_hardest_samples(CONST.dir_rgb_hardest, CONST.dir_rgb, files_to_move_rgb)

    files_to_move_texture = files_to_move(result, CONST.dir_texture)
    copy_hardest_samples(CONST.dir_texture_hardest, CONST.dir_texture, files_to_move_texture)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        print(kie)
