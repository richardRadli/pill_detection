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

from config.config import ConfigStreamNetwork
from config.config_selector import stream_network_config, sub_stream_network_configs
from utils.utils import find_latest_file_in_latest_directory, setup_logger, mine_hard_triplets


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- P R E P R O C E S S   P A T H -------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def preprocess_path(paths):
    """

    :param paths:
    :return:
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
def find_union_triplets(*lists):
    """

    :param lists:
    :return:
    """

    preprocessed_lists = [[preprocess_path(path) for path in triplet] for triplet in lists]
    union_triplets = (set(preprocessed_lists[0]) | set(preprocessed_lists[1]) |
                      set(preprocessed_lists[2]) | set(preprocessed_lists[3]))
    return union_triplets


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- G E T   H A R D E S T   S A M P L E S ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def get_hardest_samples():
    """
    The main function of the program.

    :return: None
    """

    setup_logger()
    cfg = ConfigStreamNetwork().parse()
    hard_sample_paths = stream_network_config(cfg)

    latest_hard_samples_contour = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_sample").get("Contour").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )

    latest_hard_samples_lbp = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_sample").get("LBP").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )

    latest_hard_samples_rgb = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_sample").get("RGB").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )

    latest_hard_samples_texture = (
        find_latest_file_in_latest_directory(
            path=hard_sample_paths.get("hard_sample").get("Texture").get(cfg.dataset_type),
            type_of_loss=cfg.type_of_loss_func
        )
    )

    sub_stream_cfg = sub_stream_network_configs(cfg)
    hardest_contour_triplets = mine_hard_triplets(latest_hard_samples_contour)
    hardest_lpb_triplets = mine_hard_triplets(latest_hard_samples_lbp)
    hardest_rgb_triplets = mine_hard_triplets(latest_hard_samples_rgb)
    hardest_texture_triplets = mine_hard_triplets(latest_hard_samples_texture)

    common_triplets = find_union_triplets(hardest_contour_triplets, hardest_lpb_triplets,
                                          hardest_rgb_triplets, hardest_texture_triplets)

    stream_contour_anchor = sub_stream_cfg.get("Contour").get("train").get(cfg.dataset_type).get("anchor")
    stream_contour_pos_neg = sub_stream_cfg.get("Contour").get("train").get(cfg.dataset_type).get("pos_neg")
    stream_lbp_anchor = sub_stream_cfg.get("LBP").get("train").get(cfg.dataset_type).get("anchor")
    stream_lbp_pos_neg = sub_stream_cfg.get("LBP").get("train").get(cfg.dataset_type).get("pos_neg")
    stream_rgb_anchor = sub_stream_cfg.get("RGB").get("train").get(cfg.dataset_type).get("anchor")
    stream_rgb_pos_neg = sub_stream_cfg.get("RGB").get("train").get(cfg.dataset_type).get("pos_neg")
    stream_texture_anchor = sub_stream_cfg.get("Texture").get("train").get(cfg.dataset_type).get("anchor")
    stream_texture_pos_neg = sub_stream_cfg.get("Texture").get("train").get(cfg.dataset_type).get("pos_neg")

    hardest_triplets = []
    class_id_a_p = None
    class_id_n = None

    for file_name in common_triplets:
        if cfg.dataset_type == 'ogyei':
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
                        class_id_a_p = value
                    elif idx == 1:
                        pass
                    else:
                        class_id_n = value
                else:
                    raise ValueError(f"No match for file: {file}")

        else:
            raise ValueError(f"Unknown dataset type: {cfg.dataset_type}")

        contour_anchor = (os.path.join(stream_contour_anchor, class_id_a_p, f"contour_{file_name[0]}"))
        contour_positive = (os.path.join(stream_contour_pos_neg, class_id_a_p, f"contour_{file_name[1]}"))
        contour_negative = (os.path.join(stream_contour_pos_neg, class_id_n, f"contour_{file_name[2]}"))

        lbp_anchor = (os.path.join(stream_lbp_anchor, class_id_a_p, f"lbp_{file_name[0]}"))
        lbp_positive = (os.path.join(stream_lbp_pos_neg, class_id_a_p, f"lbp_{file_name[1]}"))
        lbp_negative = (os.path.join(stream_lbp_pos_neg, class_id_n, f"lbp_{file_name[2]}"))

        rgb_anchor = (os.path.join(stream_rgb_anchor, class_id_a_p, file_name[0]))
        rgb_positive = (os.path.join(stream_rgb_pos_neg, class_id_a_p, file_name[1]))
        rgb_negative = (os.path.join(stream_rgb_pos_neg, class_id_n, file_name[2]))

        texture_anchor = (os.path.join(stream_texture_anchor, class_id_a_p, f"texture_{file_name[0]}"))
        texture_positive = (os.path.join(stream_texture_pos_neg, class_id_a_p, f"texture_{file_name[1]}"))
        texture_negative = (os.path.join(stream_texture_pos_neg, class_id_n, f"texture_{file_name[2]}"))

        hardest_triplets.append((contour_anchor, contour_positive, contour_negative,
                                 lbp_anchor, lbp_positive, lbp_negative,
                                 rgb_anchor, rgb_positive, rgb_negative,
                                 texture_anchor, texture_positive, texture_negative
                                 ))
    return hardest_triplets
