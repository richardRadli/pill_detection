"""
File: const.py
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
            "PROJECT_ROOT":
                "D:/storage/pill_detection",
            "DATASET_ROOT":
                "D:/storage/pill_detection/datasets"
        },
        "keplab": {
            "PROJECT_ROOT":
                "/home/keplab/Documents/users/radli_richard/storage/pill_detection",
            "DATASET_ROOT":
                "/home/keplab/Documents/users/radli_richard/datasets/pill_detection"
        }
    }

    if user in root_mapping:
        root_info = root_mapping[user]
        PROJECT_ROOT = root_info["PROJECT_ROOT"]
        DATASET_ROOT = root_info["DATASET_ROOT"]
    else:
        raise ValueError("Wrong user!")

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- C R E A T E   D I R C T O R I E S ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def create_directories(cls, dirs, root_type) -> None:
        """
        Class method that creates the missing directories.
        :param dirs: These are the directories that the function checks.
        :param root_type: Either PROJECT or DATASET.
        :return: None
        """

        for _, path in dirs.items():
            if root_type == "PROJECT":
                dir_path = os.path.join(cls.PROJECT_ROOT, path)
            elif root_type == "DATASET":
                dir_path = os.path.join(cls.DATASET_ROOT, path)
            else:
                raise ValueError("Wrong root type!")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Directory {dir_path} has been created")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++ I M A G E S +++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Images(_Const):
    dirs_images = {
        # +++++++++++++++++++++++++++++++++++++++++++++++++++ C U R E ++++++++++++++++++++++++++++++++++++++++++++++++++
        "stream_images_cure_two_sided_anchor":
            "images/cure_two_sided/stream_images/anchor",
        "stream_images_cure_two_sided_pos_neg":
            "images/cure_two_sided/stream_images/pos_neg",

        # ------------------------------------------------- A N C H O R ------------------------------------------------
        "train_contour_stream_cure_two_sided_anchor":
            "images/cure_two_sided/stream_images/anchor/contour",
        "train_lbp_stream_cure_two_sided_anchor":
            "images/cure_two_sided/stream_images/anchor/lbp",
        "train_rgb_stream_cure_two_sided_anchor":
            "images/cure_two_sided/stream_images/anchor/rgb",
        "train_texture_stream_cure_two_sided_anchor":
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
        "test_query_cure_two_sided":
            "images/cure_two_sided/test/query",
        "test_contour_stream_query_cure_two_sided":
            "images/cure_two_sided/test/query/contour",
        "test_lbp_stream_query_cure_two_sided":
            "images/cure_two_sided/test/query/lbp",
        "test_rgb_stream_query_cure_two_sided":
            "images/cure_two_sided/test/query/rgb",
        "test_texture_stream_query_cure_two_sided":
            "images/cure_two_sided/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "test_ref_cure_two_sided":
            "images/cure_two_sided/test/ref",
        "test_contour_stream_ref_cure_two_sided":
            "images/cure_two_sided/test/ref/contour",
        "test_lbp_stream_ref_cure_two_sided":
            "images/cure_two_sided/test/ref/lbp",
        "test_rgb_stream_ref_cure_two_sided":
            "images/cure_two_sided/test/ref/rgb",
        "test_texture_stream_ref_cure_two_sided":
            "images/cure_two_sided/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_cure_two_sided":
            "images/cure_two_sided/plotting/stream_net/plotting_efficient_net",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_multi_head_attention_cure_two_sided":
            "images/cure_two_sided/plotting/fusion_net/plotting_fusion_network_efficient_net_multi_head_attention",

        # ---------------------------- C O N F U S I O N   M A T R I X   S T R E A M   N E T ---------------------------
        "conf_mtx_efficient_net_cure_two_sided":
            "images/cure_two_sided/conf_mtx/stream_net/conf_mtx_efficient_net",

        # --------------------------- C O N F U S I O N   M A T R I X   F U S I O N   N E T ----------------------------
        "conf_mtx_fusion_network_efficient_net_multi_head_attention_cure_two_sided":
            "images/cure_two_sided/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_multi_head_attention",

        "Fourier_collected_images_by_shape_cure_two_sided":
            "images/cure_two_sided/dynamic_margin/Fourier_descriptor/collected_images_by_shape",
        "Fourier_euclidean_distance_cure_two_sided":
            "images/cure_two_sided/dynamic_margin/Fourier_descriptor/euclidean_distance",

        "combined_vectors_euc_dst_cure_two_sided":
            "images/cure_two_sided/dynamic_margin/combined_vectors_euc_dst",

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I ++++++++++++++++++++++++++++++++++++++++++++++++
        "stream_images_ogyei_anchor":
            "images/ogyei/stream_images/anchor",
        "stream_images_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg",

        # ------------------------------------------------- A N C H O R ------------------------------------------------
        "train_contour_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/contour",
        "train_lbp_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/lbp",
        "train_rgb_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/rgb",
        "train_texture_stream_ogyei_anchor":
            "images/ogyei/stream_images/anchor/texture",

        # ----------------------------------------------- P O S   N E G ------------------------------------------------
        "contour_stream_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg/contour",
        "lbp_stream_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg/lbp",
        "rgb_stream_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg/rgb",
        "texture_stream_ogyei_pos_neg":
            "images/ogyei/stream_images/pos_neg/texture",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "test_query_ogyei":
            "images/ogyei/test/query",
        "test_contour_stream_query_ogyei":
            "images/ogyei/test/query/contour",
        "test_lbp_stream_query_ogyei":
            "images/ogyei/test/query/lbp",
        "test_rgb_stream_query_ogyei":
            "images/ogyei/test/query/rgb",
        "test_texture_stream_query_ogyei":
            "images/ogyei/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "test_ref_ogyei":
            "images/ogyei/test/ref",
        "test_contour_stream_ref_ogyei":
            "images/ogyei/test/ref/contour",
        "test_lbp_stream_ref_ogyei":
            "images/ogyei/test/ref/lbp",
        "test_rgb_stream_ref_ogyei":
            "images/ogyei/test/ref/rgb",
        "test_texture_stream_ref_ogyei":
            "images/ogyei/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_ogyei":
            "images/ogyei/plotting/stream_net/plotting_efficient_net",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_multi_head_attention_ogyei":
            "images/ogyei/plotting/fusion_net/plotting_fusion_network_efficient_net_multi_head_attention",

        # ---------------------------- C O N F U S I O N   M A T R I X   S T R E A M   N E T ---------------------------
        "conf_mtx_efficient_net_ogyei":
            "images/ogyei/conf_mtx/stream_net/conf_mtx_efficient_net",

        # --------------------------- C O N F U S I O N   M A T R I X   F U S I O N   N E T ----------------------------
        "conf_mtx_fusion_network_efficient_net_multi_head_attention_ogyei":
            "images/ogyei/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_multi_head_attention",

        "Fourier_collected_images_by_shape_ogyei":
            "images/ogyei/dynamic_margin/Fourier_descriptor/collected_images_by_shape",
        "Fourier_euclidean_distance_ogyei":
            "images/ogyei/dynamic_margin/Fourier_descriptor/euclidean_distance",

        "combined_vectors_euc_dst_ogyei":
            "images/ogyei/dynamic_margin/combined_vectors_euc_dst",

        # ++++++++++++++++++++++++++++++++++++++++ C U R E   O N E   S I D E D +++++++++++++++++++++++++++++++++++++++++
        "stream_images_cure_one_sided_anchor":
            "images/cure_one_sided/stream_images/anchor",
        "stream_images_cure_one_sided_pos_neg":
            "images/cure_one_sided/stream_images/pos_neg",

        # --------------------------------------- R E F   T R A I N   A N C H O R --------------------------------------
        "train_contour_stream_cure_one_sided_anchor":
            "images/cure_one_sided/stream_images/anchor/contour",
        "train_lbp_stream_cure_one_sided_anchor":
            "images/cure_one_sided/stream_images/anchor/lbp",
        "train_rgb_stream_cure_one_sided_anchor":
            "images/cure_one_sided/stream_images/anchor/rgb",
        "train_texture_stream_cure_one_sided_anchor":
            "images/cure_one_sided/stream_images/anchor/texture",

        # -------------------------------------------- R E F   A N C H O R ---------------------------------------------
        "contour_stream_cure_one_sided_pos_neg":
            "images/cure_one_sided/stream_images/pos_neg/contour",
        "lbp_stream_cure_one_sided_pos_neg":
            "images/cure_one_sided/stream_images/pos_neg/lbp",
        "rgb_stream_cure_one_sided_pos_neg":
            "images/cure_one_sided/stream_images/pos_neg/rgb",
        "texture_stream_cure_one_sided_pos_neg":
            "images/cure_one_sided/stream_images/pos_neg/texture",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "test_query_cure_one_sided":
            "images/cure_one_sided/test/query",
        "test_contour_stream_query_cure_one_sided":
            "images/cure_one_sided/test/query/contour",
        "test_lbp_stream_query_cure_one_sided":
            "images/cure_one_sided/test/query/lbp",
        "test_rgb_stream_query_cure_one_sided":
            "images/cure_one_sided/test/query/rgb",
        "test_texture_stream_query_cure_one_sided":
            "images/cure_one_sided/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "test_ref_cure_one_sided":
            "images/cure_one_sided/test/ref",
        "test_contour_stream_ref_cure_one_sided":
            "images/cure_one_sided/test/ref/contour",
        "test_lbp_stream_ref_cure_one_sided":
            "images/cure_one_sided/test/ref/lbp",
        "test_rgb_stream_ref_cure_one_sided":
            "images/cure_one_sided/test/ref/rgb",
        "test_texture_stream_ref_cure_one_sided":
            "images/cure_one_sided/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_cure_one_sided":
            "images/cure_one_sided/plotting/stream_net/plotting_efficient_net",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_multi_head_attention_cure_one_sided":
            "images/cure_one_sided/plotting/fusion_net/plotting_fusion_network_efficient_net_multi_head_attention",

        # ---------------------------- C O N F U S I O N   M A T R I X   S T R E A M   N E T ---------------------------
        "conf_mtx_efficient_net_cure_one_sided":
            "images/cure_one_sided/conf_mtx/stream_net/conf_mtx_efficient_net",

        # --------------------------- C O N F U S I O N   M A T R I X   F U S I O N   N E T ----------------------------
        "conf_mtx_fusion_network_efficient_net_multi_head_attention_cure_one_sided":
            "images/cure_one_sided/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_multi_head_attention",

        # ++++++++++++++++++++++++++++++++++++++++++++++++++ O T H E R +++++++++++++++++++++++++++++++++++++++++++++++++
        # ----------------------------------------------- F O U R I E R ------------------------------------------------
        "Fourier_collected_images_by_shape_cure_one_sided":
            "images/cure_one_sided/dynamic_margin/Fourier_descriptor/collected_images_by_shape",
        "Fourier_euclidean_distance_cure_one_sided":
            "images/cure_one_sided/dynamic_margin/Fourier_descriptor/euclidean_distance",

        # ---------------------------------------------- C O L O U R S -------------------------------------------------
        "combined_vectors_euc_dst_cure_one_sided":
            "images/cure_one_sided/dynamic_margin/combined_vectors_euc_dst",
    }

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------- I N I T -----------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_images, "PROJECT")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_images.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Data(_Const):
    dirs_data = {
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++ C U R E   O N E   S I D E D +++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNet - StreamNetwork
        "weights_efficient_net_contour_cure_one_sided":
            "data/cure_one_sided/weights/stream_net/efficient_net/weights_efficient_net_contour",
        "weights_efficient_net_lbp_cure_one_sided":
            "data/cure_one_sided/weights/stream_net/efficient_net/weights_efficient_net_lbp",
        "weights_efficient_net_rgb_cure_one_sided":
            "data/cure_one_sided/weights/stream_net/efficient_net/weights_efficient_net_rgb",
        "weights_efficient_net_texture_cure_one_sided":
            "data/cure_one_sided/weights/stream_net/efficient_net/weights_efficient_net_texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_multi_head_attention_cure_one_sided":
            "data/cure_one_sided/weights/fusion_net/efficient_net_multi_head_attention",

        # ---------------------------------- W E I G H T S   W O R D   E M B E D D E D ---------------------------------
        "weights_word_embedded_network_cure_one_sided":
            "data/cure_one_sided/weights/word_embedded_network",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNet
        "logs_efficient_net_contour_cure_one_sided":
            "data/cure_one_sided/logs/stream_net/efficient_net/logs_efficient_net_contour",
        "logs_efficient_net_lbp_cure_one_sided":
            "data/cure_one_sided/logs/stream_net/efficient_net/logs_efficient_net_lbp",
        "logs_efficient_net_rgb_cure_one_sided":
            "data/cure_one_sided/logs/stream_net/efficient_net/logs_efficient_net_rgb",
        "logs_efficient_net_texture_cure_one_sided":
            "data/cure_one_sided/logs/stream_net/efficient_net/logs_efficient_net_texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_multi_head_attention_cure_one_sided":
            "data/cure_one_sided/logs/fusion_net/efficient_net_multi_head_attention",

        # ------------------------------------- L O G S   W O R D   E M B E D D E D ------------------------------------
        "logs_word_embedded_network_cure_one_sided":
            "data/cure_one_sided/logs/word_embedded_network",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_efficient_net_contour_cure_one_sided":
            "data/cure_one_sided/hardest_samples/efficient_net/contour",
        "hardest_samples_efficient_net_lbp_cure_one_sided":
            "data/cure_one_sided/hardest_samples/efficient_net/lbp",
        "hardest_samples_efficient_net_rgb_cure_one_sided":
            "data/cure_one_sided/hardest_samples/efficient_net/rgb",
        "hardest_samples_efficient_net_texture_cure_one_sided":
            "data/cure_one_sided/hardest_samples/efficient_net/texture",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_cure_one_sided":
            "data/cure_one_sided/predictions/stream_net/predictions_efficient_net",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_multi_head_attention_cure_one_sided":
            "data/cure_one_sided/predictions/fusion_net/predictions_fusion_network_efficient_net_multi_head_attention_net",

        # ------------------------------ P R E D I C T I O N S    W O R D   E M B E D E D ------------------------------
        # Predictions
        "predictions_word_embedded_network_cure_one_sided":
            "data/cure_one_sided/predictions/word_embedded_network",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_cure_one_sided":
            "data/cure_one_sided/ref_vec/stream_net/reference_vectors_efficient_net",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "reference_vectors_fusion_network_efficient_net_multi_head_attention_cure_one_sided":
            "data/cure_one_sided/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_multi_head_attention",

        # ---------------------------------------- D Y N A M I C   M A R G I N -----------------------------------------
        "pill_desc_xlsx_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/xlsx",
        "Fourier_saved_mean_vectors_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/Fourier_descriptor",
        "colour_vectors_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/colour_vectors",
        "imprint_vectors_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/imprint_vectors",
        "score_vectors_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/score_vectors",
        "concatenated_vectors_cure_one_sided":
            "data/cure_one_sided/dynamic_margin/concatenated_vectors",

        # ------------------------------------------------ K   F O L D -------------------------------------------------
        "cure_one_sided_k_fold":
            "data/cure_one_sided/k_fold",


        # ++++++++++++++++++++++++++++++++++++++++ C U R E   T W O   S I D E D  ++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNet - StreamNetwork
        "weights_efficient_net_contour_cure_two_sided":
            "data/cure_two_sided/weights/stream_net/efficient_net/weights_efficient_net_contour",
        "weights_efficient_net_lbp_cure_two_sided":
            "data/cure_two_sided/weights/stream_net/efficient_net/weights_efficient_net_lbp",
        "weights_efficient_net_rgb_cure_two_sided":
            "data/cure_two_sided/weights/stream_net/efficient_net/weights_efficient_net_rgb",
        "weights_efficient_net_texture_cure_two_sided":
            "data/cure_two_sided/weights/stream_net/efficient_net/weights_efficient_net_texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_multi_head_attention_cure_two_sided":
            "data/cure_two_sided/weights/fusion_net/efficient_net_multi_head_attention",

        # ---------------------------------- W E I G H T S   W O R D   E M B E D D E D ---------------------------------
        "weights_word_embedded_network_cure_two_sided":
            "data/cure_two_sided/weights/word_embedded_network",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNet 
        "logs_efficient_net_contour_cure_two_sided":
            "data/cure_two_sided/logs/stream_net/efficient_net/logs_efficient_net_contour",
        "logs_efficient_net_lbp_cure_two_sided":
            "data/cure_two_sided/logs/stream_net/efficient_net/logs_efficient_net_lbp",
        "logs_efficient_net_rgb_cure_two_sided":
            "data/cure_two_sided/logs/stream_net/efficient_net/logs_efficient_net_rgb",
        "logs_efficient_net_texture_cure_two_sided":
            "data/cure_two_sided/logs/stream_net/efficient_net/logs_efficient_net_texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_multi_head_attention_cure_two_sided":
            "data/cure_two_sided/logs/fusion_net/efficient_net_multi_head_attention",

        # ------------------------------------- L O G S   W O R D   E M B E D D E D ------------------------------------
        "logs_word_embedded_network_cure_two_sided":
            "data/cure_two_sided/logs/word_embedded_network",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_efficient_net_contour_cure_two_sided":
            "data/cure_two_sided/hardest_samples/efficient_net/contour",
        "hardest_samples_efficient_net_lbp_cure_two_sided":
            "data/cure_two_sided/hardest_samples/efficient_net/lbp",
        "hardest_samples_efficient_net_rgb_cure_two_sided":
            "data/cure_two_sided/hardest_samples/efficient_net/rgb",
        "hardest_samples_efficient_net_texture_cure_two_sided":
            "data/cure_two_sided/hardest_samples/efficient_net/texture",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_cure_two_sided":
            "data/cure_two_sided/predictions/stream_net/predictions_efficient_net",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_multi_head_attention_cure_two_sided":
            "data/cure_two_sided/predictions/fusion_net/predictions_fusion_network_efficient_net_multi_head_attention_net",

        # ------------------------------ P R E D I C T I O N S    W O R D   E M B E D E D ------------------------------
        # Predictions
        "predictions_word_embedded_network_cure_two_sided":
            "data/cure_two_sided/predictions/word_embedded_network",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_cure_two_sided":
            "data/cure_two_sided/ref_vec/stream_net/reference_vectors_efficient_net",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "reference_vectors_fusion_network_efficient_net_multi_head_attention_cure_two_sided":
            "data/cure_two_sided/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_multi_head_attention",

        # ---------------------------------------- D Y N A M I C   M A R G I N -----------------------------------------
        "pill_desc_xlsx_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/xlsx",
        "Fourier_saved_mean_vectors_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/Fourier_descriptor",
        "colour_vectors_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/colour_vectors",
        "imprint_vectors_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/imprint_vectors",
        "score_vectors_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/score_vectors",
        "concatenated_vectors_cure_two_sided":
            "data/cure_two_sided/dynamic_margin/concatenated_vectors",

        "cure_two_sided_k_fold":
            "data/cure_two_sided/k_fold",

        # ++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I +++++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNet - StreamNetwork
        "weights_efficient_net_contour_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_contour",
        "weights_efficient_net_lbp_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_lbp",
        "weights_efficient_net_rgb_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_rgb",
        "weights_efficient_net_texture_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_multi_head_attention_ogyei":
            "data/ogyei/weights/fusion_net/efficient_net_multi_head_attention",

        # ---------------------------------- W E I G H T S   W O R D   E M B E D D E D ---------------------------------
        "weights_word_embedded_network_ogyei":
            "data/ogyei/weights/word_embedded_network",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNet 
        "logs_efficient_net_contour_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_contour",
        "logs_efficient_net_lbp_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_lbp",
        "logs_efficient_net_rgb_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_rgb",
        "logs_efficient_net_texture_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_multi_head_attention_ogyei":
            "data/ogyei/logs/fusion_net/efficient_net_multi_head_attention",

        # ------------------------------------- L O G S   W O R D   E M B E D D E D ------------------------------------
        "logs_word_embedded_network_ogyei":
            "data/ogyei/logs/word_embedded_network",

        # ---------------------------------------- H A R D E S T   S A M P L E S ---------------------------------------
        "hardest_samples_efficient_net_contour_ogyei":
            "data/ogyei/hardest_samples/efficient_net/contour",
        "hardest_samples_efficient_net_lbp_ogyei":
            "data/ogyei/hardest_samples/efficient_net/lbp",
        "hardest_samples_efficient_net_rgb_ogyei":
            "data/ogyei/hardest_samples/efficient_net/rgb",
        "hardest_samples_efficient_net_texture_ogyei":
            "data/ogyei/hardest_samples/efficient_net/texture",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_ogyei":
            "data/ogyei/predictions/stream_net/predictions_efficient_net",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_multi_head_attention_ogyei":
            "data/ogyei/predictions/fusion_net/predictions_fusion_network_efficient_net_multi_head_attention_net",

        # ------------------------------ P R E D I C T I O N S    W O R D   E M B E D E D ------------------------------
        # Predictions
        "predictions_word_embedded_network_ogyei":
            "data/ogyei/predictions/word_embedded_network",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_ogyei":
            "data/ogyei/ref_vec/stream_net/reference_vectors_efficient_net",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "reference_vectors_fusion_network_efficient_net_multi_head_attention_ogyei":
            "data/ogyei/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_multi_head_attention",

        "ogyei_k_fold":
            "data/ogyei/k_fold",

        "pill_desc_xlsx_ogyei":
            "data/ogyei/dynamic_margin/xlsx",
        "Fourier_saved_mean_vectors_ogyei":
            "data/ogyei/dynamic_margin/Fourier_descriptor",
        "colour_vectors_ogyei":
            "data/ogyei/dynamic_margin/colour_vectors",
        "imprint_vectors_ogyei":
            "data/ogyei/dynamic_margin/imprint_vectors",
        "score_vectors_ogyei":
            "data/ogyei/dynamic_margin/score_vectors",
        "concatenated_vectors_ogyei":
            "data/ogyei/dynamic_margin/concatenated_vectors"
    }

    def __init__(self):
        super().__init__()
        self.create_directories(self.dirs_data, "PROJECT")

    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.dirs_data.get(key, ""))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++ D A T A S E T ++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Datasets(_Const):
    dirs_dataset = {
        # --------------------------------------------------- D T D ----------------------------------------------------
        "dtd_images":
            "dtd_images",

        # -------------------------------------------------- C U R E ---------------------------------------------------
        # CUSTOMER
        "cure_two_sided_customer_images":
            "cure_two_sided/Customer/images",
        "cure_two_sided_customer_mask_images":
            "cure_two_sided/Customer/mask_images",
        "cure_two_sided_customer_segmentation_labels":
            "cure_two_sided/Customer/segmentation_labels",

        # REFERENCE
        "cure_two_sided_reference_images":
            "cure_two_sided/Reference/images",
        "cure_two_sided_reference_mask_images":
            "cure_two_sided/Reference/mask_images",
        "cure_two_sided_reference_segmentation_labels":
            "cure_two_sided/Reference/segmentation_labels",
        "cure_two_sided_reference_yolo_labels":
            "cure_two_sided/Reference/yolo_labels",

        # CUSTOMER SPLITTED
        "cure_two_sided_train_images":
            "cure_two_sided/Customer_splitted/train_dir/images",
        "cure_two_sided_train_mask_images":
            "cure_two_sided/Customer_splitted/train_dir/mask_images",
        "cure_two_sided_train_segmentation_labels":
            "cure_two_sided/Customer_splitted/train_dir/segmentation_labels",

        "cure_two_sided_valid_images":
            "cure_two_sided/Customer_splitted/valid_dir/images",
        "cure_two_sided_valid_mask_images":
            "cure_two_sided/Customer_splitted/valid_dir/mask_images",
        "cure_two_sided_valid_segmentation_labels":
            "cure_two_sided/Customer_splitted/valid_dir/segmentation_labels",

        "cure_two_sided_test_images":
            "cure_two_sided/Customer_splitted/test_dir/images",
        "cure_two_sided_test_mask_images":
            "cure_two_sided/Customer_splitted/test_dir/mask_images",
        "cure_two_sided_test_segmentation_labels":
            "cure_two_sided/Customer_splitted/test_dir/segmentation_labels",

        # CUSTOMER SPLITTED AUGMENTED
        "cure_two_sided_train_aug_images":
            "cure_two_sided/Customer_splitted_aug/train_dir/images",
        "cure_two_sided_train_aug_yolo_labels":
            "cure_two_sided/Customer_splitted_aug/train_dir/yolo_labels",
        "cure_two_sided_train_aug_mask_images":
            "cure_two_sided/Customer_splitted_aug/train_dir/mask_images",

        "cure_two_sided_valid_aug_images":
            "cure_two_sided/Customer_splitted_aug/valid_dir/images",
        "cure_two_sided_valid_aug_yolo_labels":
            "cure_two_sided/Customer_splitted_aug/valid_dir/yolo_labels",
        "cure_two_sided_valid_aug_mask_images":
            "cure_two_sided/Customer_splitted_aug/valid_dir/mask_images",

        "cure_two_sided_test_aug_images":
            "cure_two_sided/Customer_splitted_aug/test_dir/images",
        "cure_two_sided_test_aug_yolo_labels":
            "cure_two_sided/Customer_splitted_aug/test_dir/yolo_labels",

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

        # ------------------------------------------------- O G Y E I --------------------------------------------------
        # CUSTOMER
        "ogyei_customer_images":
            "ogyei/Customer/images",
        "ogyei_customer_segmentation_labels":
            "ogyei/Customer/segmentation_labels",
        "ogyei_customer_mask_images":
            "ogyei/Customer/mask_images",

        # REFERENCE
        "ogyei_reference_images":
            "ogyei/Reference/images",
        "ogyei_reference_segmentation_labels":
            "ogyei/Reference/segmentation_labels",
        "ogyei_reference_mask_images":
            "ogyei/Reference/mask_images",

        # CUSTOMER SPLITTED
        "ogyei_train_images":
            "ogyei/Customer_splitted/train_dir/images",
        "ogyei_train_mask_images":
            "ogyei/Customer_splitted/train_dir/mask_images",
        "ogyei_train_segmentation_labels":
            "ogyei/Customer_splitted/train_dir/segmentation_labels",

        "ogyei_valid_images":
            "ogyei/Customer_splitted/valid_dir/images",
        "ogyei_valid_mask_images":
            "ogyei/Customer_splitted/valid_dir/mask_images",
        "ogyei_valid_segmentation_labels":
            "ogyei/Customer_splitted/valid_dir/segmentation_labels",

        "ogyei_test_images":
            "ogyei/Customer_splitted/test_dir/images",
        "ogyei_test_mask_images":
            "ogyei/Customer_splitted/test_dir/mask_images",
        "ogyei_test_segmentation_labels":
            "ogyei/Customer_splitted/test_dir/segmentation_labels",

        # STREAM - Customer
        "stream_images_ogyei_customer":
            "ogyei/Customer/stream_images",
        "stream_images_ogyei_customer_contour":
            "ogyei/Customer/stream_images/contour",
        "stream_images_ogyei_customer_lbp":
            "ogyei/Customer/stream_images/lbp",
        "stream_images_ogyei_customer_rgb":
            "ogyei/Customer/stream_images/rgb",
        "stream_images_ogyei_customer_texture":
            "ogyei/Customer/stream_images/texture",

        # STREAM - Reference
        "stream_images_ogyei_reference":
            "ogyei/Reference/stream_images",
        "stream_images_ogyei_reference_contour":
            "ogyei/Reference/stream_images/contour",
        "stream_images_ogyei_reference_lbp":
            "ogyei/Reference/stream_images/lbp",
        "stream_images_ogyei_reference_rgb":
            "ogyei/Reference/stream_images/rgb",
        "stream_images_ogyei_reference_texture":
            "ogyei/Reference/stream_images/texture",

        # ---------------------------------------- C U R E    O N E   S I D E D ----------------------------------------
        # CUSTOMER
        "cure_one_sided_customer_images":
            "cure_one_sided/Customer/images",
        "cure_one_sided_customer_mask_images":
            "cure_one_sided/Customer/mask_images",
        "cure_one_sided_customer_segmentation_labels":
            "cure_one_sided/Customer/segmentation_labels",

        # REFERENCE
        "cure_one_sided_reference_images":
            "cure_one_sided/Reference/images",
        "cure_one_sided_reference_mask_images":
            "cure_one_sided/Reference/mask_images",
        "cure_one_sided_reference_segmentation_labels":
            "cure_one_sided/Reference/segmentation_labels",
        "cure_one_sided_reference_yolo_labels":
            "cure_one_sided/Reference/yolo_labels",

        # CUSTOMER SPLITTED
        "cure_one_sided_train_images":
            "cure_one_sided/Customer_splitted/train_dir/images",
        "cure_one_sided_train_mask_images":
            "cure_one_sided/Customer_splitted/train_dir/mask_images",
        "cure_one_sided_train_segmentation_labels":
            "cure_one_sided/Customer_splitted/train_dir/segmentation_labels",

        "cure_one_sided_valid_images":
            "cure_one_sided/Customer_splitted/valid_dir/images",
        "cure_one_sided_valid_mask_images":
            "cure_one_sided/Customer_splitted/valid_dir/mask_images",
        "cure_one_sided_valid_segmentation_labels":
            "cure_one_sided/Customer_splitted/valid_dir/segmentation_labels",

        "cure_one_sided_test_images":
            "cure_one_sided/Customer_splitted/test_dir/images",
        "cure_one_sided_test_mask_images":
            "cure_one_sided/Customer_splitted/test_dir/mask_images",
        "cure_one_sided_test_segmentation_labels":
            "cure_one_sided/Customer_splitted/test_dir/segmentation_labels",

        # CUSTOMER SPLITTED AUGMENTED
        "cure_one_sided_train_aug_images":
            "cure_one_sided/Customer_splitted_aug/train_dir/images",
        "cure_one_sided_train_aug_yolo_labels":
            "cure_one_sided/Customer_splitted_aug/train_dir/yolo_labels",
        "cure_one_sided_train_aug_mask_images":
            "cure_one_sided/Customer_splitted_aug/train_dir/mask_images",

        "cure_one_sided_valid_aug_images":
            "cure_one_sided/Customer_splitted_aug/valid_dir/images",
        "cure_one_sided_valid_aug_yolo_labels":
            "cure_one_sided/Customer_splitted_aug/valid_dir/yolo_labels",
        "cure_one_sided_valid_aug_mask_images":
            "cure_one_sided/Customer_splitted_aug/valid_dir/mask_images",

        "cure_one_sided_test_aug_images":
            "cure_one_sided/Customer_splitted_aug/test_dir/images",
        "cure_one_sided_test_aug_yolo_labels":
            "cure_one_sided/Customer_splitted_aug/test_dir/yolo_labels",

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
            "cure_one_sided/Reference/stream_images/texture"
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
        self.create_directories(self.nlp_data, "PROJECT")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ G E T   D A T A   P A T H ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def get_data_path(self, key):
        return os.path.join(self.PROJECT_ROOT, self.nlp_data.get(key, ""))


CONST: _Const = _Const()
IMAGES_PATH: Images = Images()
DATA_PATH: Data = Data()
DATASET_PATH: Datasets = Datasets()
NLP_DATA_PATH: NLPData = NLPData()
