"""
File: mine_hard_samples.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Feb 06, 2024

Description:
This program collects the hard samples (images) that were mined during the stream network phase.
"""

import os.path
import re

from typing import List, Tuple

from config.json_config import json_config_selector
from config.networks_paths_selector import stream_network_backbone_paths,  substream_paths
from utils.utils import find_latest_file_in_latest_directory, setup_logger, mine_hard_triplets, load_config_json


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- P R E P R O C E S S   P A T H -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def preprocess_path(paths: List[str]) -> tuple:
    """
    Preprocesses a list of file paths.

    Args:
        paths (List[str]): List of file paths.

    Returns:
        tuple: Preprocessed paths.
    """

    preprocessed_paths = []
    for path in paths:
        parts = path.replace('\\', '/').split('/')
        basename = parts[-1]

        for prefix in ['contour_', 'lbp_', 'texture_']:
            if basename.startswith(prefix):
                basename = basename[len(prefix):]
                break
        preprocessed_paths.append(basename)

    return tuple(preprocessed_paths)


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- F I N D   T R I P L E T S ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def find_union_triplets(*lists: [List[Tuple[str, str, str]]]) -> set:
    """
    Finds the union of triplets from multiple lists.

    Args:
        *lists (List[Tuple[str, str, str]]): Variable number of lists containing triplets.

    Returns:
        set: Union of all triplets.
    """

    preprocessed_lists = [[preprocess_path(path) for path in triplet] for triplet in lists]
    union_triplets = (set(preprocessed_lists[0]) | set(preprocessed_lists[1]) |
                      set(preprocessed_lists[2]) | set(preprocessed_lists[3]))
    return union_triplets


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ M A I N -------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_hardest_samples():
    """
    The main function of the program.

    Returns:
         None
    """

    setup_logger()
    cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("stream_net").get("schema"),
                json_filename=json_config_selector("stream_net").get("config")
            )
        )

    dataset_type = cfg.get("dataset_type")
    network_type = cfg.get("type_of_net")
    loss_type = cfg.get("type_of_loss_func")
    dmtl_type = cfg.get("type_of_dmtl")

    hard_sample_paths = (
            stream_network_backbone_paths(
                dataset_type=dataset_type,
                network_type=network_type
            )
        )

    if loss_type == "hmtl":
        hard_sample_path_by_loss = hard_sample_paths.get("hard_sample").get(loss_type)
    else:
        hard_sample_path_by_loss = hard_sample_paths.get("hard_sample").get(loss_type).get(dmtl_type)

    latest_hard_samples_contour = (
        find_latest_file_in_latest_directory(
            path=hard_sample_path_by_loss.get("Contour")
        )
    )

    latest_hard_samples_lbp = (
        find_latest_file_in_latest_directory(
            path=hard_sample_path_by_loss.get("LBP")
        )
    )

    latest_hard_samples_rgb = (
        find_latest_file_in_latest_directory(
            path=hard_sample_path_by_loss.get("RGB")
        )
    )

    latest_hard_samples_texture = (
        find_latest_file_in_latest_directory(
            path=hard_sample_path_by_loss.get("Texture")
        )
    )

    hardest_contour_triplets = mine_hard_triplets(latest_hard_samples_contour)
    hardest_lpb_triplets = mine_hard_triplets(latest_hard_samples_lbp)
    hardest_rgb_triplets = mine_hard_triplets(latest_hard_samples_rgb)
    hardest_texture_triplets = mine_hard_triplets(latest_hard_samples_texture)

    common_triplets = find_union_triplets(hardest_contour_triplets, hardest_lpb_triplets,
                                          hardest_rgb_triplets, hardest_texture_triplets)

    stream_contour_anchor = (
        substream_paths().get("Contour").get(dataset_type).get(network_type).get("train").get("anchor")
    )
    stream_contour_pos_neg = (
        substream_paths().get("Contour").get(dataset_type).get(network_type).get("train").get("pos_neg")
    )
    stream_lbp_anchor = (
        substream_paths().get("LBP").get(dataset_type).get(network_type).get("train").get("anchor")
    )
    stream_lbp_pos_neg = (
        substream_paths().get("LBP").get(dataset_type).get(network_type).get("train").get("pos_neg")
    )
    stream_rgb_anchor = (
        substream_paths().get("RGB").get(dataset_type).get(network_type).get("train").get("anchor")
    )
    stream_rgb_pos_neg = (
        substream_paths().get("RGB").get(dataset_type).get(network_type).get("train").get("pos_neg")
    )
    stream_texture_anchor = (
        substream_paths().get("Texture").get(dataset_type).get(network_type).get("train").get("anchor")
    )
    stream_texture_pos_neg = (
        substream_paths().get("Texture").get(dataset_type).get(network_type).get("train").get("pos_neg")
    )

    hardest_triplets = []
    class_id_a = None
    class_id_p = None
    class_id_n = None

    for file_name in common_triplets:

        if dataset_type == "cure_two_sided":
            class_id_a = file_name[0].split("_")[0]
            class_id_p = class_id_a
            class_id_n = file_name[2].split("_")[0]

        elif dataset_type == 'ogyei':
            for idx, file in enumerate(file_name):
                if "_s_" in file:
                    match = re.search(r'^(.*?)_s_\d{3}\.jpg$', file)
                elif "_u_" in file:
                    match = re.search(r'^(.*?)_u_\d{3}\.jpg$', file)
                else:
                    raise ValueError(f"Unrecognized file: {file}")
                if match:
                    value = match.group(1)
                    if idx == 0:
                        class_id_a = value
                        class_id_p = class_id_a
                    elif idx == 1:
                        pass
                    else:
                        class_id_n = value
                else:
                    raise ValueError(f"No match for file: {file}")

        elif dataset_type == "cure_one_sided":
            class_id_a = file_name[0].split("_")[0] + "_" + file_name[0].split("_")[1]
            class_id_p = file_name[1].split("_")[0] + "_" + file_name[1].split("_")[2].split(".")[0]
            class_id_n = file_name[2].split("_")[0] + "_" + file_name[2].split("_")[2].split(".")[0]

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        assert class_id_a == class_id_p

        contour_anchor = (os.path.join(stream_contour_anchor, class_id_a, f"contour_{file_name[0]}"))
        contour_positive = (os.path.join(stream_contour_pos_neg, class_id_p, f"contour_{file_name[1]}"))
        contour_negative = (os.path.join(stream_contour_pos_neg, class_id_n, f"contour_{file_name[2]}"))

        lbp_anchor = (os.path.join(stream_lbp_anchor, class_id_a, f"lbp_{file_name[0]}"))
        lbp_positive = (os.path.join(stream_lbp_pos_neg, class_id_p, f"lbp_{file_name[1]}"))
        lbp_negative = (os.path.join(stream_lbp_pos_neg, class_id_n, f"lbp_{file_name[2]}"))

        rgb_anchor = (os.path.join(stream_rgb_anchor, class_id_a, file_name[0]))
        rgb_positive = (os.path.join(stream_rgb_pos_neg, class_id_p, file_name[1]))
        rgb_negative = (os.path.join(stream_rgb_pos_neg, class_id_n, file_name[2]))

        texture_anchor = (os.path.join(stream_texture_anchor, class_id_a, f"texture_{file_name[0]}"))
        texture_positive = (os.path.join(stream_texture_pos_neg, class_id_p, f"texture_{file_name[1]}"))
        texture_negative = (os.path.join(stream_texture_pos_neg, class_id_n, f"texture_{file_name[2]}"))

        hardest_triplets.append((contour_anchor, contour_positive, contour_negative,
                                 lbp_anchor, lbp_positive, lbp_negative,
                                 rgb_anchor, rgb_positive, rgb_negative,
                                 texture_anchor, texture_positive, texture_negative
                                 ))
    return hardest_triplets
