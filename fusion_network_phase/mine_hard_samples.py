"""
File: mine_hard_samples.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description:
This program collects the hard samples (images) that were mined during the stream network phase.
"""

import logging
import os
import shutil

from tqdm import tqdm

from config.config import ConfigStreamNetwork
from config.config_selector import stream_network_config, sub_stream_network_configs
from utils.utils import find_latest_file_in_latest_directory, setup_logger


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
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    """
    The main function of the program.

    :return: None
    """

    setup_logger()
    cfg = ConfigStreamNetwork().parse()
    hard_sample_paths = stream_network_config(cfg)
    ref_dir_paths = sub_stream_network_configs(cfg)

    latest_neg_contour_txt = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_negative").get("contour").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )
    latest_neg_rgb_txt = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_negative").get("rgb").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )
    latest_neg_texture_txt = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_negative").get("texture").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )
    latest_neg_lbp_txt = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_negative").get("lbp").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )

    latest_pos_contour_txt = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_positive").get("contour").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )
    latest_pos_rgb_txt = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_positive").get("rgb").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )
    latest_pos_texture_txt = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_positive").get("texture").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )
    latest_pos_lbp_txt = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_positive").get("lbp").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )

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
    files_to_move_contour = (
        files_to_move(hardest_sample_images=result,
                      src_dir=ref_dir_paths.get("Contour").get("train").get(cfg.dataset_type))
    )

    copy_hardest_samples(
        new_dir=hard_sample_paths.get("hardest_contour_directory").get(cfg.dataset_type),
        src_dir=ref_dir_paths.get("Contour").get("train").get(cfg.dataset_type),
        hardest_sample_images=files_to_move_contour
    )

    # Move hardest lbp images
    files_to_move_lbp = (
        files_to_move(hardest_sample_images=result,
                      src_dir=ref_dir_paths.get("LBP").get("train").get(cfg.dataset_type))
    )

    copy_hardest_samples(
        new_dir=hard_sample_paths.get("hardest_lbp_directory").get(cfg.dataset_type),
        src_dir=ref_dir_paths.get("LBP").get("train").get(cfg.dataset_type),
        hardest_sample_images=files_to_move_lbp
    )

    # Move hardest rgb images
    files_to_move_rgb = (
        files_to_move(hardest_sample_images=result,
                      src_dir=ref_dir_paths.get("RGB").get("train").get(cfg.dataset_type))
    )
    copy_hardest_samples(
        new_dir=hard_sample_paths.get("hardest_rgb_directory").get(cfg.dataset_type),
        src_dir=ref_dir_paths.get("RGB").get("train").get(cfg.dataset_type),
        hardest_sample_images=files_to_move_rgb
    )

    # Move hardest texture images
    files_to_move_texture = (
        files_to_move(hardest_sample_images=result,
                      src_dir=ref_dir_paths.get("Texture").get("train").get(cfg.dataset_type))
    )

    copy_hardest_samples(
        new_dir=hard_sample_paths.get("hardest_texture_directory").get(cfg.dataset_type),
        src_dir=ref_dir_paths.get("Texture").get("train").get(cfg.dataset_type),
        hardest_sample_images=files_to_move_texture
    )


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- __M A I N__ -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        logging.error("Keyboard interrupt has happened!")
