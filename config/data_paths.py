"""
File: data_paths.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program stores the const values of different variables. There is a main class, named _Const(), and 3
other classes are inherited from that (Images, Data, Dataset).
"""

import logging
import os

from utils.utils import setup_logger


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N S T ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class _Const(object):
    # Setup logger
    setup_logger()

    # Select user and according paths
    user = os.getlogin()
    root_mapping = {
        "ricsi": {
            "STORAGE_ROOT":
                "D:/storage/pill_detection/wiley",
            "DATASET_ROOT":
                "D:/storage/pill_detection/wiley/datasets",
            "PROJECT_ROOT":
                "C:/Users/ricsi/Documents/project/IVM",
        }
    }

    if user in root_mapping:
        root_info = root_mapping[user]
        STORAGE_ROOT = root_info["STORAGE_ROOT"]
        DATASET_ROOT = root_info["DATASET_ROOT"]
        PROJECT_ROOT = root_info["PROJECT_ROOT"]
    else:
        raise ValueError("Wrong user!")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   D I R C T O R I E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def create_directories(cls, dirs, root_type) -> None:
        """
        Class method that creates the missing directories.

        Args:
            dirs: These are the directories that the function checks.
            root_type: Either STORAGE or DATASET.

        Returns:
             None
        """

        for _, path in dirs.items():
            if root_type == "STORAGE":
                dir_path = os.path.join(cls.STORAGE_ROOT, path)
            elif root_type == "PROJECT":
                dir_path = os.path.join(cls.PROJECT_ROOT, path)
            elif root_type == "DATASET":
                dir_path = os.path.join(cls.DATASET_ROOT, path)
            else:
                raise ValueError("Wrong root type!")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Directory {dir_path} has been created")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++ C O N F I G +++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ConfigFilePaths(_Const):
    dirs_config_paths = {
        "config_augmentation":
            "config/json_files/augmentation_config.json",
        "config_schema_augmentation":
            "config/json_files/augmentation_config_schema.json",

        "config_fusion_net":
            "config/json_files/fusion_net_config.json",
        "config_schema_fusion_net":
            "config/json_files/fusion_net_config_schema.json",

        "config_stream_images":
            "config/json_files/stream_images_config.json",
        "config_schema_stream_images":
            "config/json_files/stream_images_config_schema.json",

        "config_stream_net":
            "config/json_files/stream_net_config.json",
        "config_schema_stream_net":
            "config/json_files/stream_net_config_schema.json",

        "config_word_embedding":
            "config/json_files/word_embedding_config.json",
        "config_schema_word_embedding":
            "config/json_files/word_embedding_config_schema.json"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_config_paths.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++ I M A G E S +++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Images(_Const):
    dirs_images = {
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I ++++++++++++++++++++++++++++++++++++++++++++++++
        "stream_images_ogyei_v2_anchor":
            "images/ogyei_v2/stream_images/anchor",
        "stream_images_ogyei_v2_pos_neg":
            "images/ogyei_v2/stream_images/pos_neg",

        # ------------------------------------------------- A N C H O R ------------------------------------------------
        "contour_stream_ogyei_v2_anchor":
            "images/ogyei_v2/stream_images/anchor/contour",
        "lbp_stream_ogyei_v2_anchor":
            "images/ogyei_v2/stream_images/anchor/lbp",
        "rgb_stream_ogyei_v2_anchor":
            "images/ogyei_v2/stream_images/anchor/rgb",
        "texture_stream_ogyei_v2_anchor":
            "images/ogyei_v2/stream_images/anchor/texture",

        # ----------------------------------------------- P O S   N E G ------------------------------------------------
        "contour_stream_ogyei_v2_pos_neg":
            "images/ogyei_v2/stream_images/pos_neg/contour",
        "lbp_stream_ogyei_v2_pos_neg":
            "images/ogyei_v2/stream_images/pos_neg/lbp",
        "rgb_stream_ogyei_v2_pos_neg":
            "images/ogyei_v2/stream_images/pos_neg/rgb",
        "texture_stream_ogyei_v2_pos_neg":
            "images/ogyei_v2/stream_images/pos_neg/texture",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "query_ogyei_v2":
            "images/ogyei_v2/test/query",
        "contour_stream_query_ogyei_v2":
            "images/ogyei_v2/test/query/contour",
        "lbp_stream_query_ogyei_v2":
            "images/ogyei_v2/test/query/lbp",
        "rgb_stream_query_ogyei_v2":
            "images/ogyei_v2/test/query/rgb",
        "texture_stream_query_ogyei_v2":
            "images/ogyei_v2/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "ref_ogyei_v2":
            "images/ogyei_v2/test/ref",
        "contour_stream_ref_ogyei_v2":
            "images/ogyei_v2/test/ref/contour",
        "lbp_stream_ref_ogyei_v2":
            "images/ogyei_v2/test/ref/lbp",
        "rgb_stream_ref_ogyei_v2":
            "images/ogyei_v2/test/ref/rgb",
        "texture_stream_ref_ogyei_v2":
            "images/ogyei_v2/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_v2_ogyei_v2_hmtl":
            "images/ogyei_v2/plotting/stream_net/efficient_net_v2/hmtl",
        "plotting_efficient_net_v2_ogyei_v2_dmtl_visual":
            "images/ogyei_v2/plotting/stream_net/efficient_net_v2/dmtl_visual",
        "plotting_efficient_net_v2_ogyei_v2_dmtl_textual":
            "images/ogyei_v2/plotting/stream_net/efficient_net_v2/dmtl_textual",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl":
            "images/ogyei_v2/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/hmtl",
        "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_visual":
            "images/ogyei_v2/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/dmtl_visual",
        "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_textual":
            "images/ogyei_v2/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/dmtl_textual",

        # -------------------------------- E M B E D D I N G   E U C L I D E A N   M T X -------------------------------
        "emb_euc_mtx_word_embedded_network_ogyei_v2":
            "images/ogyei_v2/plotting/word_embedded_network/emb_euc_mtx",

        # ---------------------------------------- E M B E D D I N G   T S N E -----------------------------------------
        "emb_tsne_word_embedded_network_ogyei_v2":
            "images/ogyei_v2/plotting/word_embedded_network/emb_tsne_vis",

        # ------------------------------------------- F O U R I E R ----------------------------------------------------
        "Fourier_euclidean_distance_ogyei_v2":
            "images/ogyei_v2/dynamic_margin/Fourier/euclidean_distance",
        "Fourier_collected_images_by_shape_ogyei_v2":
            "images/ogyei_v2/dynamic_margin/Fourier/collected_images",
        "combined_vectors_euc_dst_ogyei_v2":
            "images/ogyei_v2/dynamic_margin/Fourier/combined_vectors_euc_dst",

        # ++++++++++++++++++++++++++++++++++++++++++ C U R E   O N E   S I D E D +++++++++++++++++++++++++++++++++++++++
        "stream_images_cure_one_sided_anchor":
            "images/cure_one_sided/stream_images/anchor",
        "stream_images_cure_one_sided_pos_neg":
            "images/cure_one_sided/stream_images/pos_neg",

        # ------------------------------------------------- A N C H O R ------------------------------------------------
        "contour_stream_cure_one_sided_anchor":
            "images/cure_one_sided/stream_images/anchor/contour",
        "lbp_stream_cure_one_sided_anchor":
            "images/cure_one_sided/stream_images/anchor/lbp",
        "rgb_stream_cure_one_sided_anchor":
            "images/cure_one_sided/stream_images/anchor/rgb",
        "texture_stream_cure_one_sided_anchor":
            "images/cure_one_sided/stream_images/anchor/texture",

        # ----------------------------------------------- P O S   N E G ------------------------------------------------
        "contour_stream_cure_one_sided_pos_neg":
            "images/cure_one_sided/stream_images/pos_neg/contour",
        "lbp_stream_cure_one_sided_pos_neg":
            "images/cure_one_sided/stream_images/pos_neg/lbp",
        "rgb_stream_cure_one_sided_pos_neg":
            "images/cure_one_sided/stream_images/pos_neg/rgb",
        "texture_stream_cure_one_sided_pos_neg":
            "images/cure_one_sided/stream_images/pos_neg/texture",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "query_cure_one_sided":
            "images/cure_one_sided/test/query",
        "contour_stream_query_cure_one_sided":
            "images/cure_one_sided/test/query/contour",
        "lbp_stream_query_cure_one_sided":
            "images/cure_one_sided/test/query/lbp",
        "rgb_stream_query_cure_one_sided":
            "images/cure_one_sided/test/query/rgb",
        "texture_stream_query_cure_one_sided":
            "images/cure_one_sided/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "ref_cure_one_sided":
            "images/cure_one_sided/test/ref",
        "contour_stream_ref_cure_one_sided":
            "images/cure_one_sided/test/ref/contour",
        "lbp_stream_ref_cure_one_sided":
            "images/cure_one_sided/test/ref/lbp",
        "rgb_stream_ref_cure_one_sided":
            "images/cure_one_sided/test/ref/rgb",
        "texture_stream_ref_cure_one_sided":
            "images/cure_one_sided/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_v2_cure_one_sided_hmtl":
            "images/cure_one_sided/plotting/stream_net/efficient_net_v2/hmtl",
        "plotting_efficient_net_v2_cure_one_sided_dmtl_visual":
            "images/cure_one_sided/plotting/stream_net/efficient_net_v2/dmtl_visual",
        "plotting_efficient_net_v2_cure_one_sided_dmtl_textual":
            "images/cure_one_sided/plotting/stream_net/efficient_net_v2/dmtl_textual",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_hmtl":
            "images/cure_one_sided/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/hmtl",
        "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_visual":
            "images/cure_one_sided/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/dmtl_visual",
        "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_textual":
            "images/cure_one_sided/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/dmtl_textual",

        # -------------------------------- E M B E D D I N G   E U C L I D E A N   M T X -------------------------------
        "emb_euc_mtx_word_embedded_network_cure_one_sided":
            "images/cure_one_sided/plotting/word_embedded_network/emb_euc_mtx",

        # ---------------------------------------- E M B E D D I N G   T S N E -----------------------------------------
        "emb_tsne_word_embedded_network_cure_one_sided":
            "images/cure_one_sided/plotting/word_embedded_network/emb_tsne_vis",

        # ------------------------------------------- F O U R I E R ----------------------------------------------------
        "Fourier_euclidean_distance_cure_one_sided":
            "images/cure_one_sided/dynamic_margin/Fourier/euclidean_distance",
        "Fourier_collected_images_by_shape_cure_one_sided":
            "images/cure_one_sided/dynamic_margin/Fourier/collected_images",
        "combined_vectors_euc_dst_cure_one_sided":
            "images/cure_one_sided/dynamic_margin/Fourier/combined_vectors_euc_dst",

        # ++++++++++++++++++++++++++++++++++++++++++ C U R E   T W O   S I D E D +++++++++++++++++++++++++++++++++++++++
        "stream_images_cure_two_sided_anchor":
            "images/cure_two_sided/stream_images/anchor",
        "stream_images_cure_two_sided_pos_neg":
            "images/cure_two_sided/stream_images/pos_neg",

        # ------------------------------------------------- A N C H O R ------------------------------------------------
        "contour_stream_cure_two_sided_anchor":
            "images/cure_two_sided/stream_images/anchor/contour",
        "lbp_stream_cure_two_sided_anchor":
            "images/cure_two_sided/stream_images/anchor/lbp",
        "rgb_stream_cure_two_sided_anchor":
            "images/cure_two_sided/stream_images/anchor/rgb",
        "texture_stream_cure_two_sided_anchor":
            "images/cure_two_sided/stream_images/anchor/texture",

        # ----------------------------------------------- P O S   N E G ------------------------------------------------
        "contour_stream_cure_two_sided_pos_neg":
            "images/cure_two_sided/stream_images/pos_neg/contour",
        "lbp_stream_cure_two_sided_pos_neg":
            "images/cure_two_sided/stream_images/pos_neg/lbp",
        "rgb_stream_cure_two_sided_pos_neg":
            "images/cure_two_sided/stream_images/pos_neg/rgb",
        "texture_stream_cure_two_sided_pos_neg":
            "images/cure_two_sided/stream_images/pos_neg/texture",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "query_cure_two_sided":
            "images/cure_two_sided/test/query",
        "contour_stream_query_cure_two_sided":
            "images/cure_two_sided/test/query/contour",
        "lbp_stream_query_cure_two_sided":
            "images/cure_two_sided/test/query/lbp",
        "rgb_stream_query_cure_two_sided":
            "images/cure_two_sided/test/query/rgb",
        "texture_stream_query_cure_two_sided":
            "images/cure_two_sided/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "ref_cure_two_sided":
            "images/cure_two_sided/test/ref",
        "contour_stream_ref_cure_two_sided":
            "images/cure_two_sided/test/ref/contour",
        "lbp_stream_ref_cure_two_sided":
            "images/cure_two_sided/test/ref/lbp",
        "rgb_stream_ref_cure_two_sided":
            "images/cure_two_sided/test/ref/rgb",
        "texture_stream_ref_cure_two_sided":
            "images/cure_two_sided/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_v2_cure_two_sided_hmtl":
            "images/cure_two_sided/plotting/stream_net/efficient_net_v2/hmtl",
        "plotting_efficient_net_v2_cure_two_sided_dmtl_visual":
            "images/cure_two_sided/plotting/stream_net/efficient_net_v2/dmtl_visual",
        "plotting_efficient_net_v2_cure_two_sided_dmtl_textual":
            "images/cure_two_sided/plotting/stream_net/efficient_net_v2/dmtl_textual",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_hmtl":
            "images/cure_two_sided/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/hmtl",
        "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_visual":
            "images/cure_two_sided/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/dmtl_visual",
        "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_textual":
            "images/cure_two_sided/plotting/fusion_net/fusion_network_efficient_net_v2_multihead_attention/dmtl_textual",

        # -------------------------------- E M B E D D I N G   E U C L I D E A N   M T X -------------------------------
        "emb_euc_mtx_word_embedded_network_cure_two_sided":
            "images/cure_two_sided/plotting/word_embedded_network/emb_euc_mtx",

        # ---------------------------------------- E M B E D D I N G   T S N E -----------------------------------------
        "emb_tsne_word_embedded_network_cure_two_sided":
            "images/cure_two_sided/plotting/word_embedded_network/emb_tsne_vis",

        # ------------------------------------------- F O U R I E R ----------------------------------------------------
        "Fourier_euclidean_distance_cure_two_sided":
            "images/cure_two_sided/dynamic_margin/Fourier/euclidean_distance",
        "Fourier_collected_images_by_shape_cure_two_sided":
            "images/cure_two_sided/dynamic_margin/Fourier/collected_images",
        "combined_vectors_euc_dst_cure_two_sided":
            "images/cure_two_sided/dynamic_margin/Fourier/combined_vectors_euc_dst",
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_images, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_images.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Data(_Const):
    dirs_data = {
        # ++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I +++++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNetV2 - StreamNetwork
        "weights_efficient_net_v2_contour_ogyei_v2_hmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/contour/hmtl",
        "weights_efficient_net_v2_lbp_ogyei_v2_hmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/lbp/hmtl",
        "weights_efficient_net_v2_rgb_ogyei_v2_hmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/rgb/hmtl",
        "weights_efficient_net_v2_texture_ogyei_v2_hmtl":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/texture/hmtl",

        "weights_efficient_net_v2_contour_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/contour/dmtl_visual",
        "weights_efficient_net_v2_lbp_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/lbp/dmtl_visual",
        "weights_efficient_net_v2_rgb_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/rgb/dmtl_visual",
        "weights_efficient_net_v2_texture_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/texture/dmtl_visual",

        "weights_efficient_net_v2_contour_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/contour/dmtl_textual",
        "weights_efficient_net_v2_lbp_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/lbp/dmtl_textual",
        "weights_efficient_net_v2_rgb_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/rgb/dmtl_textual",
        "weights_efficient_net_v2_texture_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/weights/stream_net/efficient_net_v2/texture/dmtl_textual",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl":
            "data/ogyei_v2/weights/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/weights/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/weights/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNetV2
        "logs_efficient_net_v2_contour_ogyei_v2_hmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/contour/hmtl",
        "logs_efficient_net_v2_lbp_ogyei_v2_hmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/lbp/hmtl",
        "logs_efficient_net_v2_rgb_ogyei_v2_hmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/rgb/hmtl",
        "logs_efficient_net_v2_texture_ogyei_v2_hmtl":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/texture/hmtl",

        "logs_efficient_net_v2_contour_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/contour/dmtl_visual",
        "logs_efficient_net_v2_lbp_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/lbp/dmtl_visual",
        "logs_efficient_net_v2_rgb_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/rgb/dmtl_visual",
        "logs_efficient_net_v2_texture_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/texture/dmtl_visual",

        "logs_efficient_net_v2_contour_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/contour/dmtl_textual",
        "logs_efficient_net_v2_lbp_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/lbp/dmtl_textual",
        "logs_efficient_net_v2_rgb_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/rgb/dmtl_textual",
        "logs_efficient_net_v2_texture_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/logs/stream_net/efficient_net_v2/texture/dmtl_textual",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl":
            "data/ogyei_v2/logs/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "logs_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/logs/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "logs_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/logs/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_v2_ogyei_v2_hmtl":
            "data/ogyei_v2/predictions/stream_net/efficient_net_v2/hmtl",
        "predictions_efficient_net_v2_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/predictions/stream_net/efficient_net_v2/dmtl_visual",
        "predictions_efficient_net_v2_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/predictions/stream_net/efficient_net_v2/dmtl_textual",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl":
            "data/ogyei_v2/predictions/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/predictions/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/predictions/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_v2_ogyei_v2_hmtl":
            "data/ogyei_v2/ref_vec/stream_net/efficient_net_v2/hmtl",
        "reference_vectors_efficient_net_v2_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/ref_vec/stream_net/efficient_net_v2/dmtl_visual",
        "reference_vectors_efficient_net_v2_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/ref_vec/stream_net/efficient_net_v2/dmtl_textual",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl":
            "data/ogyei_v2/ref_vec/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/ref_vec/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/ref_vec/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_efficient_net_v2_contour_ogyei_v2_hmtl":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/contour/hmtl",
        "hardest_samples_efficient_net_v2_lbp_ogyei_v2_hmtl":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/lbp/hmtl",
        "hardest_samples_efficient_net_v2_rgb_ogyei_v2_hmtl":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/rgb/hmtl",
        "hardest_samples_efficient_net_v2_texture_ogyei_v2_hmtl":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/texture/hmtl",

        "hardest_samples_efficient_net_v2_contour_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/contour/dmtl_visual",
        "hardest_samples_efficient_net_v2_lbp_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/lbp/dmtl_visual",
        "hardest_samples_efficient_net_v2_rgb_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/rgb/dmtl_visual",
        "hardest_samples_efficient_net_v2_texture_ogyei_v2_dmtl_visual":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/texture/dmtl_visual",

        "hardest_samples_efficient_net_v2_contour_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/contour/dmtl_textual",
        "hardest_samples_efficient_net_v2_lbp_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/lbp/dmtl_textual",
        "hardest_samples_efficient_net_v2_rgb_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/rgb/dmtl_textual",
        "hardest_samples_efficient_net_v2_texture_ogyei_v2_dmtl_textual":
            "data/ogyei_v2/hardest_samples/efficient_net_v2/texture/dmtl_textual",

        # ------------------------------------------------ K   F O L D -------------------------------------------------
        "ogyei_v2_k_fold":
            "data/ogyei_v2/k_fold",

        # ---------------------------------- W E I G H T S   W O R D   E M B E D D E D ---------------------------------
        "weights_word_embedded_network_ogyei_v2":
            "data/ogyei_v2/weights/word_embedded_network",
        # ------------------------------------- L O G S   W O R D   E M B E D D E D ------------------------------------
        "logs_word_embedded_network_ogyei_v2":
            "data/ogyei_v2/logs/word_embedded_network",
        # ------------------------------ P R E D I C T I O N S    W O R D   E M B E D E D ------------------------------
        # Predictions
        "predictions_word_embedded_network_ogyei_v2":
            "data/ogyei_v2/predictions/word_embedded_network",

        # ----------------------------------------- D Y N A M I C   M A R G I N ----------------------------------------
        "pill_desc_xlsx_ogyei_v2":
            "data/ogyei_v2/dynamic_margin/pill_desc_xlsx",
        "Fourier_saved_mean_vectors_ogyei_v2":
            "data/ogyei_v2/dynamic_margin/Fourier_saved_mean_vectors",
        "colour_vectors_ogyei_v2":
            "data/ogyei_v2/dynamic_margin/colour_vectors",
        "imprint_vectors_ogyei_v2":
            "data/ogyei_v2/dynamic_margin/imprint_vectors",
        "score_vectors_ogyei_v2":
            "data/ogyei_v2/dynamic_margin/score_vectors",
        "concatenated_vectors_ogyei_v2":
            "data/ogyei_v2/dynamic_margin/concatenated_vectors",
        "euc_mtx_xlsx_ogyei_v2":
            "data/ogyei_v2/dynamic_margin/euc_mtx_xlsx",

        # +++++++++++++++++++++++++++++++++++++++++++++++ C U R E   O N E ++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNetV2 - StreamNetwork
        "weights_efficient_net_v2_contour_cure_one_sided_hmtl":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/contour/hmtl",
        "weights_efficient_net_v2_lbp_cure_one_sided_hmtl":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/lbp/hmtl",
        "weights_efficient_net_v2_rgb_cure_one_sided_hmtl":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/rgb/hmtl",
        "weights_efficient_net_v2_texture_cure_one_sided_hmtl":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/texture/hmtl",

        "weights_efficient_net_v2_contour_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/contour/dmtl_visual",
        "weights_efficient_net_v2_lbp_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/lbp/dmtl_visual",
        "weights_efficient_net_v2_rgb_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/rgb/dmtl_visual",
        "weights_efficient_net_v2_texture_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/texture/dmtl_visual",

        "weights_efficient_net_v2_contour_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/contour/dmtl_textual",
        "weights_efficient_net_v2_lbp_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/lbp/dmtl_textual",
        "weights_efficient_net_v2_rgb_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/rgb/dmtl_textual",
        "weights_efficient_net_v2_texture_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/weights/stream_net/efficient_net_v2/texture/dmtl_textual",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_hmtl":
            "data/cure_one_sided/weights/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "weights_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/weights/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "weights_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/weights/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNetV2
        "logs_efficient_net_v2_contour_cure_one_sided_hmtl":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/contour/hmtl",
        "logs_efficient_net_v2_lbp_cure_one_sided_hmtl":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/lbp/hmtl",
        "logs_efficient_net_v2_rgb_cure_one_sided_hmtl":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/rgb/hmtl",
        "logs_efficient_net_v2_texture_cure_one_sided_hmtl":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/texture/hmtl",

        "logs_efficient_net_v2_contour_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/contour/dmtl_visual",
        "logs_efficient_net_v2_lbp_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/lbp/dmtl_visual",
        "logs_efficient_net_v2_rgb_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/rgb/dmtl_visual",
        "logs_efficient_net_v2_texture_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/texture/dmtl_visual",

        "logs_efficient_net_v2_contour_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/contour/dmtl_textual",
        "logs_efficient_net_v2_lbp_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/lbp/dmtl_textual",
        "logs_efficient_net_v2_rgb_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/rgb/dmtl_textual",
        "logs_efficient_net_v2_texture_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/logs/stream_net/efficient_net_v2/texture/dmtl_textual",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_hmtl":
            "data/cure_one_sided/logs/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "logs_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/logs/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "logs_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/logs/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_v2_cure_one_sided_hmtl":
            "data/cure_one_sided/predictions/stream_net/efficient_net_v2/hmtl",
        "predictions_efficient_net_v2_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/predictions/stream_net/efficient_net_v2/dmtl_visual",
        "predictions_efficient_net_v2_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/predictions/stream_net/efficient_net_v2/dmtl_textual",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_hmtl":
            "data/cure_one_sided/predictions/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/predictions/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/predictions/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_v2_cure_one_sided_hmtl":
            "data/cure_one_sided/ref_vec/stream_net/efficient_net_v2/hmtl",
        "reference_vectors_efficient_net_v2_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/ref_vec/stream_net/efficient_net_v2/dmtl_visual",
        "reference_vectors_efficient_net_v2_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/ref_vec/stream_net/efficient_net_v2/dmtl_textual",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_hmtl":
            "data/cure_one_sided/ref_vec/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/ref_vec/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/ref_vec/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_efficient_net_v2_contour_cure_one_sided_hmtl":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/contour/hmtl",
        "hardest_samples_efficient_net_v2_lbp_cure_one_sided_hmtl":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/lbp/hmtl",
        "hardest_samples_efficient_net_v2_rgb_cure_one_sided_hmtl":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/rgb/hmtl",
        "hardest_samples_efficient_net_v2_texture_cure_one_sided_hmtl":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/texture/hmtl",

        "hardest_samples_efficient_net_v2_contour_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/contour/dmtl_visual",
        "hardest_samples_efficient_net_v2_lbp_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/lbp/dmtl_visual",
        "hardest_samples_efficient_net_v2_rgb_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/rgb/dmtl_visual",
        "hardest_samples_efficient_net_v2_texture_cure_one_sided_dmtl_visual":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/texture/dmtl_visual",

        "hardest_samples_efficient_net_v2_contour_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/contour/dmtl_textual",
        "hardest_samples_efficient_net_v2_lbp_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/lbp/dmtl_textual",
        "hardest_samples_efficient_net_v2_rgb_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/rgb/dmtl_textual",
        "hardest_samples_efficient_net_v2_texture_cure_one_sided_dmtl_textual":
            "data/cure_one_sided/hardest_samples/efficient_net_v2/texture/dmtl_textual",

        "cure_one_sided_k_fold":
            "data/cure_one_sided/k_fold",

        # ---------------------------------- W E I G H T S   W O R D   E M B E D D E D ---------------------------------
        "weights_word_embedded_network_cure_one_sided":
            "data/cure_one_sided/weights/word_embedded_network",
        # ------------------------------------- L O G S   W O R D   E M B E D D E D ------------------------------------
        "logs_word_embedded_network_cure_one_sided":
            "data/cure_one_sided/logs/word_embedded_network",
        # ------------------------------ P R E D I C T I O N S    W O R D   E M B E D E D ------------------------------
        # Predictions
        "predictions_word_embedded_network_cure_one_sided":
            "data/cure_one_sided/predictions/word_embedded_network",

        # ----------------------------------------- D Y N A M I C   M A R G I N ----------------------------------------
        "pill_desc_xlsx_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/pill_desc_xlsx",
        "Fourier_saved_mean_vectors_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/Fourier_saved_mean_vectors",
        "colour_vectors_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/colour_vectors",
        "imprint_vectors_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/imprint_vectors",
        "score_vectors_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/score_vectors",
        "concatenated_vectors_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/concatenated_vectors",
        "euc_mtx_xlsx_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/euc_mtx_xlsx",

        # +++++++++++++++++++++++++++++++++++++++++++++++ C U R E   T W O ++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNetV2 - StreamNetwork
        "weights_efficient_net_v2_contour_cure_two_sided_hmtl":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/contour/hmtl",
        "weights_efficient_net_v2_lbp_cure_two_sided_hmtl":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/lbp/hmtl",
        "weights_efficient_net_v2_rgb_cure_two_sided_hmtl":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/rgb/hmtl",
        "weights_efficient_net_v2_texture_cure_two_sided_hmtl":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/texture/hmtl",

        "weights_efficient_net_v2_contour_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/contour/dmtl_visual",
        "weights_efficient_net_v2_lbp_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/lbp/dmtl_visual",
        "weights_efficient_net_v2_rgb_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/rgb/dmtl_visual",
        "weights_efficient_net_v2_texture_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/texture/dmtl_visual",

        "weights_efficient_net_v2_contour_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/contour/dmtl_textual",
        "weights_efficient_net_v2_lbp_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/lbp/dmtl_textual",
        "weights_efficient_net_v2_rgb_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/rgb/dmtl_textual",
        "weights_efficient_net_v2_texture_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/weights/stream_net/efficient_net_v2/texture/dmtl_textual",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_hmtl":
            "data/cure_two_sided/weights/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "weights_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/weights/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "weights_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/weights/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNetV2
        "logs_efficient_net_v2_contour_cure_two_sided_hmtl":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/contour/hmtl",
        "logs_efficient_net_v2_lbp_cure_two_sided_hmtl":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/lbp/hmtl",
        "logs_efficient_net_v2_rgb_cure_two_sided_hmtl":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/rgb/hmtl",
        "logs_efficient_net_v2_texture_cure_two_sided_hmtl":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/texture/hmtl",

        "logs_efficient_net_v2_contour_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/contour/dmtl_visual",
        "logs_efficient_net_v2_lbp_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/lbp/dmtl_visual",
        "logs_efficient_net_v2_rgb_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/rgb/dmtl_visual",
        "logs_efficient_net_v2_texture_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/texture/dmtl_visual",

        "logs_efficient_net_v2_contour_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/contour/dmtl_textual",
        "logs_efficient_net_v2_lbp_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/lbp/dmtl_textual",
        "logs_efficient_net_v2_rgb_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/rgb/dmtl_textual",
        "logs_efficient_net_v2_texture_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/logs/stream_net/efficient_net_v2/texture/dmtl_textual",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_hmtl":
            "data/cure_two_sided/logs/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "logs_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/logs/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "logs_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/logs/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_v2_cure_two_sided_hmtl":
            "data/cure_two_sided/predictions/stream_net/efficient_net_v2/hmtl",
        "predictions_efficient_net_v2_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/predictions/stream_net/efficient_net_v2/dmtl_visual",
        "predictions_efficient_net_v2_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/predictions/stream_net/efficient_net_v2/dmtl_textual",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_hmtl":
            "data/cure_two_sided/predictions/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/predictions/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/predictions/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_v2_cure_two_sided_hmtl":
            "data/cure_two_sided/ref_vec/stream_net/efficient_net_v2/hmtl",
        "reference_vectors_efficient_net_v2_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/ref_vec/stream_net/efficient_net_v2/dmtl_visual",
        "reference_vectors_efficient_net_v2_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/ref_vec/stream_net/efficient_net_v2/dmtl_textual",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_hmtl":
            "data/cure_two_sided/ref_vec/fusion_net/efficient_net_v2_multihead_attention/hmtl",
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/ref_vec/fusion_net/efficient_net_v2_multihead_attention/dmtl_visual",
        "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/ref_vec/fusion_net/efficient_net_v2_multihead_attention/dmtl_textual",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_efficient_net_v2_contour_cure_two_sided_hmtl":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/contour/hmtl",
        "hardest_samples_efficient_net_v2_lbp_cure_two_sided_hmtl":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/lbp/hmtl",
        "hardest_samples_efficient_net_v2_rgb_cure_two_sided_hmtl":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/rgb/hmtl",
        "hardest_samples_efficient_net_v2_texture_cure_two_sided_hmtl":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/texture/hmtl",

        "hardest_samples_efficient_net_v2_contour_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/contour/dmtl_visual",
        "hardest_samples_efficient_net_v2_lbp_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/lbp/dmtl_visual",
        "hardest_samples_efficient_net_v2_rgb_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/rgb/dmtl_visual",
        "hardest_samples_efficient_net_v2_texture_cure_two_sided_dmtl_visual":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/texture/dmtl_visual",

        "hardest_samples_efficient_net_v2_contour_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/contour/dmtl_textual",
        "hardest_samples_efficient_net_v2_lbp_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/lbp/dmtl_textual",
        "hardest_samples_efficient_net_v2_rgb_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/rgb/dmtl_textual",
        "hardest_samples_efficient_net_v2_texture_cure_two_sided_dmtl_textual":
            "data/cure_two_sided/hardest_samples/efficient_net_v2/texture/dmtl_textual",

        "cure_two_sided_k_fold":
            "data/cure_two_sided/k_fold",

        # ---------------------------------- W E I G H T S   W O R D   E M B E D D E D ---------------------------------
        "weights_word_embedded_network_cure_two_sided":
            "data/cure_two_sided/weights/word_embedded_network",
        # ------------------------------------- L O G S   W O R D   E M B E D D E D ------------------------------------
        "logs_word_embedded_network_cure_two_sided":
            "data/cure_two_sided/logs/word_embedded_network",
        # ------------------------------ P R E D I C T I O N S    W O R D   E M B E D E D ------------------------------
        # Predictions
        "predictions_word_embedded_network_cure_two_sided":
            "data/cure_two_sided/predictions/word_embedded_network",

        # ----------------------------------------- D Y N A M I C   M A R G I N ----------------------------------------
        "pill_desc_xlsx_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/pill_desc_xlsx",
        "Fourier_saved_mean_vectors_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/Fourier_saved_mean_vectors",
        "colour_vectors_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/colour_vectors",
        "imprint_vectors_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/imprint_vectors",
        "score_vectors_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/score_vectors",
        "concatenated_vectors_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/concatenated_vectors",
        "euc_mtx_xlsx_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/euc_mtx_xlsx",

    }

    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data, "STORAGE")

    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.dirs_data.get(key, ""))


class NLPData(_Const):
    nlp_data = {
        "pill_names":
            "nlp/csv/pill_names",
        "full_sentence_csv":
            "nlp/csv/full_sentence_csv",
        "vector_distances":
            "nlp/csv/distances",

        "nlp_vector":
            "nlp/npy/nlp_vector",

        "word_vector_vis":
            "nlp/plot/word_vector",
        "elbow":
            "nlp/plot/elbow",
        "silhouette":
            "nlp/plot/silhouette",

        "patient_information_leaflet_doc":
            "nlp/documents/patient_information_leaflet_doc",
        "patient_information_leaflet_docx":
            "nlp/documents/patient_information_leaflet_docx",
        "extracted_features_files":
            "nlp/documents/extracted_features_files"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.nlp_data, "STORAGE")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.STORAGE_ROOT, self.nlp_data.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Datasets(_Const):
    dirs_dataset = {
        # ------------------------------------------------- O G Y E I --------------------------------------------------
        # CUSTOMER
        "ogyei_v2_customer_images":
            "ogyei_v2/Customer/images",
        "ogyei_v2_customer_segmentation_labels":
            "ogyei_v2/Customer/segmentation_labels",
        "ogyei_v2_customer_mask_images":
            "ogyei_v2/Customer/mask_images",

        # REFERENCE
        "ogyei_v2_reference_images":
            "ogyei_v2/Reference/images",
        "ogyei_v2_reference_segmentation_labels":
            "ogyei_v2/Reference/segmentation_labels",
        "ogyei_v2_reference_mask_images":
            "ogyei_v2/Reference/mask_images",

        # STREAM - Customer
        "stream_images_ogyei_v2_customer":
            "ogyei_v2/Customer/stream_images",
        "stream_images_ogyei_v2_customer_contour":
            "ogyei_v2/Customer/stream_images/contour",
        "stream_images_ogyei_v2_customer_lbp":
            "ogyei_v2/Customer/stream_images/lbp",
        "stream_images_ogyei_v2_customer_rgb":
            "ogyei_v2/Customer/stream_images/rgb",
        "stream_images_ogyei_v2_customer_texture":
            "ogyei_v2/Customer/stream_images/texture",

        # STREAM - Reference
        "stream_images_ogyei_v2_reference":
            "ogyei_v2/Reference/stream_images",
        "stream_images_ogyei_v2_reference_contour":
            "ogyei_v2/Reference/stream_images/contour",
        "stream_images_ogyei_v2_reference_lbp":
            "ogyei_v2/Reference/stream_images/lbp",
        "stream_images_ogyei_v2_reference_rgb":
            "ogyei_v2/Reference/stream_images/rgb",
        "stream_images_ogyei_v2_reference_texture":
            "ogyei_v2/Reference/stream_images/texture",
        "stream_images_ogyei_v2_reference_masks":
            "ogyei_v2/Reference/stream_images/masks",

        # UNSPLITTED
        "ogyei_v2_images":
            "ogyei_v2/unsplitted/images",
        "ogyei_v2_mask_images":
            "ogyei_v2/unsplitted/gt_masks",
        "ogyei_v2_segmentation_labels":
            "ogyei_v2/unsplitted/labels",

        # SPLITTED
        "ogyei_v2_train_images":
            "ogyei_v2/splitted/train/images",
        "ogyei_v2_train_mask_images":
            "ogyei_v2/splitted/train/gt_train_masks",
        "ogyei_v2_train_segmentation_labels":
            "ogyei_v2/splitted/train/labels",

        "ogyei_v2_valid_images":
            "ogyei_v2/splitted/valid/images",
        "ogyei_v2_valid_mask_images":
            "ogyei_v2/splitted/valid/gt_valid_masks",
        "ogyei_v2_valid_segmentation_labels":
            "ogyei_v2/splitted/valid/labels",

        "ogyei_v2_test_images":
            "ogyei_v2/splitted/test/images",
        "ogyei_v2_test_mask_images":
            "ogyei_v2/splitted/test/gt_test_masks",
        "ogyei_v2_test_segmentation_labels":
            "ogyei_v2/splitted/test/labels",

        # ----------------------------------------- C U R E   O N E   S I D E D ----------------------------------------
        # CUSTOMER
        "cure_one_sided_customer_images":
            "cure_one_sided/Customer/images",
        "cure_one_sided_customer_segmentation_labels":
            "cure_one_sided/Customer/segmentation_labels",
        "cure_one_sided_customer_mask_images":
            "cure_one_sided/Customer/mask_images",

        # REFERENCE
        "cure_one_sided_reference_images":
            "cure_one_sided/Reference/images",
        "cure_one_sided_reference_segmentation_labels":
            "cure_one_sided/Reference/segmentation_labels",
        "cure_one_sided_reference_mask_images":
            "cure_one_sided/Reference/mask_images",

        # STREAM - Customer
        "stream_images_cure_one_sided_customer":
            "cure_one_sided/Customer/stream_images",
        "stream_images_cure_one_sided_customer_contour":
            "cure_one_sided/Customer/stream_images/contour",
        "stream_images_cure_one_sided_customer_lbp":
            "cure_one_sided/Customer/stream_images/lbp",
        "stream_images_cure_one_sided_customer_rgb":
            "cure_one_sided/Customer/stream_images/rgb",
        "stream_images_cure_one_sided_customer_texture":
            "cure_one_sided/Customer/stream_images/texture",

        # STREAM - Reference
        "stream_images_cure_one_sided_reference":
            "cure_one_sided/Reference/stream_images",
        "stream_images_cure_one_sided_reference_contour":
            "cure_one_sided/Reference/stream_images/contour",
        "stream_images_cure_one_sided_reference_lbp":
            "cure_one_sided/Reference/stream_images/lbp",
        "stream_images_cure_one_sided_reference_rgb":
            "cure_one_sided/Reference/stream_images/rgb",
        "stream_images_cure_one_sided_reference_texture":
            "cure_one_sided/Reference/stream_images/texture",
        "stream_images_cure_one_sided_reference_masks":
            "cure_one_sided/Reference/stream_images/masks",

        # ----------------------------------------- C U R E   T W O   S I D E D ----------------------------------------
        # CUSTOMER
        "cure_two_sided_customer_images":
            "cure_two_sided/Customer/images",
        "cure_two_sided_customer_segmentation_labels":
            "cure_two_sided/Customer/segmentation_labels",
        "cure_two_sided_customer_mask_images":
            "cure_two_sided/Customer/mask_images",

        # REFERENCE
        "cure_two_sided_reference_images":
            "cure_two_sided/Reference/images",
        "cure_two_sided_reference_segmentation_labels":
            "cure_two_sided/Reference/segmentation_labels",
        "cure_two_sided_reference_mask_images":
            "cure_two_sided/Reference/mask_images",

        # STREAM - Customer
        "stream_images_cure_two_sided_customer":
            "cure_two_sided/Customer/stream_images",
        "stream_images_cure_two_sided_customer_contour":
            "cure_two_sided/Customer/stream_images/contour",
        "stream_images_cure_two_sided_customer_lbp":
            "cure_two_sided/Customer/stream_images/lbp",
        "stream_images_cure_two_sided_customer_rgb":
            "cure_two_sided/Customer/stream_images/rgb",
        "stream_images_cure_two_sided_customer_texture":
            "cure_two_sided/Customer/stream_images/texture",

        # STREAM - Reference
        "stream_images_cure_two_sided_reference":
            "cure_two_sided/Reference/stream_images",
        "stream_images_cure_two_sided_reference_contour":
            "cure_two_sided/Reference/stream_images/contour",
        "stream_images_cure_two_sided_reference_lbp":
            "cure_two_sided/Reference/stream_images/lbp",
        "stream_images_cure_two_sided_reference_rgb":
            "cure_two_sided/Reference/stream_images/rgb",
        "stream_images_cure_two_sided_reference_texture":
            "cure_two_sided/Reference/stream_images/texture",
        "stream_images_cure_two_sided_reference_masks":
            "cure_two_sided/Reference/stream_images/masks",

        # CUSTOMER SPLITTED AUGMENTED
        # Train
        "cure_train_aug_images":
            "cure/Augmented/train_dir/images",
        "cure_train_aug_yolo_labels":
            "cure/Augmented/train_dir/yolo_labels",
        "cure_train_aug_mask_images":
            "cure/Augmented/train_dir/mask_images",

        # Valid
        "cure_valid_aug_images":
            "cure/Augmented/valid_dir/images",
        "cure_valid_aug_yolo_labels":
            "cure/Augmented/valid_dir/yolo_labels",
        "cure_valid_aug_mask_images":
            "cure/Augmented/valid_dir/mask_images",
        # Test
        "cure_test_images":
            "cure/Augmented/test_dir/images",
        "cure_test_yolo_labels":
            "cure/Augmented/test_dir/yolo_labels",
        "cure_test_mask_images":
            "cure/Augmented/test_dir/masks",

        "dtd":
            "dtd_images"
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_dataset, "DATASET")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.DATASET_ROOT, self.dirs_dataset.get(key, ""))


CONST: _Const = _Const()
JSON_FILES_PATHS: ConfigFilePaths = ConfigFilePaths()
IMAGES_PATH: Images = Images()
NLP_DATA_PATH: NLPData = NLPData()
DATA_PATH: Data = Data()
DATASET_PATH: Datasets = Datasets()
