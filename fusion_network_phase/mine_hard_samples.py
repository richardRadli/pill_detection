import logging
import os
import shutil

from glob import glob
from tqdm import tqdm

from config.const import DATA_PATH, IMAGES_PATH
from config.config import ConfigStreamNetwork
from config.logger_setup import setup_logger


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- F I N D   S T R E A M   F O L D E R S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_stream_folders(path):
    """

    :param path:
    :return:
    """

    found_paths = []

    dirs = sorted(glob(os.path.join(path, '????-??-??_??-??-??')), reverse=True)
    subdir_dict = {'RGB': [], 'Contour': [], 'Texture': [], 'LBP': []}

    for d in dirs:
        subdirs = ['RGB', 'Contour', 'Texture', 'LBP']
        for subdir in subdirs:
            if os.path.isdir(os.path.join(d, subdir)):
                subdir_dict[subdir].append(d)
                break

        if all(subdir_dict.values()):
            break

    for subdir, dirs in subdir_dict.items():
        logging.info(f"{subdir} directories:")
        for d in dirs:
            logging.info(f"  {d}")
            found_paths.append(d)

    return found_paths


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
            copy_of_file = copy_of_file.replace("contour_", "").replace("texture_", "").replace("lbp_", "")
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

    for src_paths in tqdm(hardest_sample_images, total=len(hardest_sample_images), desc=os.path.basename(new_dir)):
        logging.info(f"{os.path.basename(src_paths)} has been moved!")
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
def get_latest_txt_files(path: str):
    """
    Retrieves the latest .txt files for contour, RGB, and texture data in the specified directories
    for either positive or negative operations.

    :return: A tuple of the paths to the latest contour, LBP, RGB, and texture .txt files, respectively.
    """

    sub_folders = find_stream_folders(path)

    latest_rgb_txt = get_hardest_samples(path, os.path.join(os.path.basename(sub_folders[0]), "RGB"))
    latest_contour_txt = get_hardest_samples(path, os.path.join(os.path.basename(sub_folders[1]), "Contour"))
    latest_texture_txt = get_hardest_samples(path, os.path.join(os.path.basename(sub_folders[2]), "Texture"))
    latest_lbp_txt = get_hardest_samples(path, os.path.join(os.path.basename(sub_folders[3]), "lbp"))

    return latest_contour_txt, latest_rgb_txt, latest_texture_txt, latest_lbp_txt


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- G E T   P A T H S --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_paths(network_type):
    """

    :param network_type:
    :return:
    """

    hard_samples_paths = {
        'CNN': {
            'hard_negative':
                DATA_PATH.get_data_path("negative_cnn_network"),
            'hard_positive':
                DATA_PATH.get_data_path("positive_cnn_network"),
            'contour_hardest':
                IMAGES_PATH.get_data_path("contour_hardest_cnn_network"),
            "lbp_hardest":
                IMAGES_PATH.get_data_path("lbp_hardest_cnn_network"),
            "rgb_hardest":
                IMAGES_PATH.get_data_path("rgb_hardest_cnn_network"),
            "texture_hardest":
                IMAGES_PATH.get_data_path("texture_hardest_cnn_network")
        },
        'EfficientNet': {
            'hard_negative':
                DATA_PATH.get_data_path("negative_efficient_net"),
            'hard_positive':
                DATA_PATH.get_data_path("positive_efficient_net"),
            'contour_hardest':
                IMAGES_PATH.get_data_path("contour_hardest_efficient_net"),
            "lbp_hardest":
                IMAGES_PATH.get_data_path("lbp_hardest_efficient_net"),
            "rgb_hardest":
                IMAGES_PATH.get_data_path("rgb_hardest_efficient_net"),
            "texture_hardest":
                IMAGES_PATH.get_data_path("texture_hardest_efficient_net")
        },
        "EfficientNetV2": {
            'hard_negative':
                DATA_PATH.get_data_path("negative_efficient_net_v2"),
            'hard_positive':
                DATA_PATH.get_data_path("positive_efficient_net_v2"),
            'contour_hardest':
                IMAGES_PATH.get_data_path("contour_hardest_efficient_net_v2"),
            "lbp_hardest":
                IMAGES_PATH.get_data_path("lbp_hardest_efficient_net_v2"),
            "rgb_hardest":
                IMAGES_PATH.get_data_path("rgb_hardest_efficient_net_v2"),
            "texture_hardest":
                IMAGES_PATH.get_data_path("texture_hardest_efficient_net_v2")
        }
    }

    if network_type not in hard_samples_paths:
        raise ValueError(f'Invalid network type: {network_type}')

    return hard_samples_paths[network_type]


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    """

    :return:
    """

    setup_logger()
    cfg = ConfigStreamNetwork().parse()
    hard_sample_paths = get_paths(cfg.type_of_net)

    latest_neg_contour_txt, latest_neg_rgb_txt, latest_neg_texture_txt, latest_neg_lbp_txt = \
        get_latest_txt_files(hard_sample_paths.get("hard_negative"))
    latest_pos_contour_txt, latest_pos_rgb_txt, latest_pos_texture_txt, latest_pos_lbp_txt = \
        get_latest_txt_files(hard_sample_paths.get("hard_positive"))

    hardest_neg_samples_contour = process_txt(latest_neg_contour_txt)
    hardest_neg_samples_rgb = process_txt(latest_neg_rgb_txt)
    hardest_neg_samples_texture = process_txt(latest_neg_texture_txt)
    hardest_neg_sample_lbp = process_txt(latest_neg_lbp_txt)

    hardest_pos_samples_contour = process_txt(latest_pos_contour_txt)
    hardest_pos_samples_rgb = process_txt(latest_pos_rgb_txt)
    hardest_pos_samples_texture = process_txt(latest_pos_texture_txt)
    hardest_pos_sample_lbp = process_txt(latest_pos_lbp_txt)

    hardest_neg_samples_union = \
        hardest_neg_samples_contour | hardest_neg_samples_rgb | hardest_neg_samples_texture | hardest_neg_sample_lbp

    hardest_pos_samples_union = \
        hardest_pos_samples_contour | hardest_pos_samples_rgb | hardest_pos_samples_texture | hardest_pos_sample_lbp

    hardest_samples_union = hardest_pos_samples_union | hardest_neg_samples_union

    result = {os.path.basename(x) for x in hardest_samples_union}

    # Move hardest contour images
    files_to_move_contour = files_to_move(result, IMAGES_PATH.get_data_path("ref_train_contour"))
    copy_hardest_samples(new_dir=hard_sample_paths.get("contour_hardest"),
                         src_dir=IMAGES_PATH.get_data_path("ref_train_contour"),
                         hardest_sample_images=files_to_move_contour)

    # Move hardest lbp images
    files_to_move_lbp = files_to_move(result, IMAGES_PATH.get_data_path("ref_train_lbp"))
    copy_hardest_samples(new_dir=hard_sample_paths.get("lbp_hardest"),
                         src_dir=IMAGES_PATH.get_data_path("ref_train_lbp"),
                         hardest_sample_images=files_to_move_lbp)

    # Move hardest rgb images
    files_to_move_rgb = files_to_move(result, IMAGES_PATH.get_data_path("ref_train_rgb"))
    copy_hardest_samples(new_dir=hard_sample_paths.get("rgb_hardest"),
                         src_dir=IMAGES_PATH.get_data_path("ref_train_rgb"),
                         hardest_sample_images=files_to_move_rgb)

    # Move hardest texture images
    files_to_move_texture = files_to_move(result, IMAGES_PATH.get_data_path("ref_train_texture"))
    copy_hardest_samples(new_dir=hard_sample_paths.get("texture_hardest"),
                         src_dir=IMAGES_PATH.get_data_path("ref_train_texture"),
                         hardest_sample_images=files_to_move_texture)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        logging.error("Keyboard interrupt has happened!")
