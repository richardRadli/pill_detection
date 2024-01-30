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
                "D:/storage/IVM",
            "DATASET_ROOT":
                "D:/storage/IVM/datasets"
        },
        "keplab": {
            "PROJECT_ROOT":
                "/home/keplab/Documents/users/radli_richard/storage/IVM",
            "DATASET_ROOT":
                "/home/keplab/Documents/users/radli_richard/datasets/IVM"
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
        "stream_images_cure":
            "images/cure/stream_images",
        "stream_images_cure_test":
            "images/cure/test",

        # ---------------------------------------------- R E F   T R A I N ---------------------------------------------
        "train_contour_stream_cure":
            "images/cure/stream_images/contour/train",
        "train_lbp_stream_cure":
            "images/cure/stream_images/lbp/train",
        "train_rgb_stream_cure":
            "images/cure/stream_images/rgb/train",
        "train_texture_stream_cure":
            "images/cure/stream_images/texture/train",

        # ---------------------------------------------- R E F   V A L I D ---------------------------------------------
        "valid_contour_stream_cure":
            "images/cure/stream_images/contour/valid",
        "valid_lbp_stream_cure":
            "images/cure/stream_images/lbp/valid",
        "valid_rgb_stream_cure":
            "images/cure/stream_images/rgb/valid",
        "valid_texture_stream_cure":
            "images/cure/stream_images/texture/valid",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "test_query_cure":
            "images/cure/test/query",
        "test_contour_stream_query_cure":
            "images/cure/test/query/contour",
        "test_lbp_stream_query_cure":
            "images/cure/test/query/lbp",
        "test_rgb_stream_query_cure":
            "images/cure/test/query/rgb",
        "test_texture_stream_query_cure":
            "images/cure/test/query/texture",

        # ---------------------------------------------------- R E F ---------------------------------------------------
        "test_ref_cure":
            "images/cure/test/ref",
        "test_contour_stream_ref_cure":
            "images/cure/test/ref/contour",
        "test_lbp_stream_ref_cure":
            "images/cure/test/ref/lbp",
        "test_rgb_stream_ref_cure":
            "images/cure/test/ref/rgb",
        "test_texture_stream_ref_cure":
            "images/cure/test/ref/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_cure":
            "images/cure/plotting/stream_net/plotting_efficient_net",
        "plotting_efficient_net_v2_cure":
            "images/cure/plotting/stream_net/plotting_efficient_net_v2",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_self_attention_cure":
            "images/cure/plotting/fusion_net/plotting_fusion_network_efficient_net_self_attention",
        "plotting_fusion_network_efficient_net_v2_multi_head_attention_cure":
            "images/cure/plotting/fusion_net/plotting_fusion_network_efficient_net_v2_multi_head_attention",

        # ---------------------------- C O N F U S I O N   M A T R I X   S T R E A M   N E T ---------------------------
        "conf_mtx_efficient_net_cure":
            "images/cure/conf_mtx/stream_net/conf_mtx_efficient_net",
        "conf_mtx_efficient_net_v2_cure":
            "images/cure/conf_mtx/stream_net/conf_mtx_efficient_net_v2",

        # --------------------------- C O N F U S I O N   M A T R I X   F U S I O N   N E T ----------------------------
        "conf_mtx_fusion_network_efficient_net_self_attention_cure":
            "images/cure/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_self_attention",
        "conf_mtx_fusion_network_efficient_net_v2_multi_head_attention_cure":
            "images/ogyei/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_v2_multi_head_attention",


        # +++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I ++++++++++++++++++++++++++++++++++++++++++++++++
        "stream_images_ogyei":
            "images/ogyei/stream_images",
        "stream_images_ogyei_test":
            "images/ogyei/test",

        # ---------------------------------------------- R E F   T R A I N ---------------------------------------------
        "train_contour_stream_ogyei":
            "images/ogyei/stream_images/contour/train",
        "train_lbp_stream_ogyei":
            "images/ogyei/stream_images/lbp/train",
        "train_rgb_stream_ogyei":
            "images/ogyei/stream_images/rgb/train",
        "train_texture_stream_ogyei":
            "images/ogyei/stream_images/texture/train",

        # ---------------------------------------------- R E F   V A L I D ---------------------------------------------
        "valid_contour_stream_ogyei":
            "images/ogyei/stream_images/contour/valid",
        "valid_lbp_stream_ogyei":
            "images/ogyei/stream_images/lbp/valid",
        "valid_rgb_stream_ogyei":
            "images/ogyei/stream_images/rgb/valid",
        "valid_texture_stream_ogyei":
            "images/ogyei/stream_images/texture/valid",

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
        "plotting_efficient_net_v2_ogyei":
            "images/ogyei/plotting/stream_net/plotting_efficient_net_v2",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_self_attention_ogyei":
            "images/ogyei/plotting/fusion_net/plotting_fusion_network_efficient_net_self_attention",
        "plotting_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "images/ogyei/plotting/fusion_net/plotting_fusion_network_efficient_net_v2_multi_head_attention",

        # ---------------------------- C O N F U S I O N   M A T R I X   S T R E A M   N E T ---------------------------
        "conf_mtx_efficient_net_ogyei":
            "images/ogyei/conf_mtx/stream_net/conf_mtx_efficient_net",
        "conf_mtx_efficient_net_v2_ogyei":
            "images/ogyei/conf_mtx/stream_net/conf_mtx_efficient_net_v2",

        # --------------------------- C O N F U S I O N   M A T R I X   F U S I O N   N E T ----------------------------
        "conf_mtx_fusion_network_efficient_net_self_attention_ogyei":
            "images/ogyei/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_self_attention",
        "conf_mtx_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "images/ogyei/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_v2_multi_head_attention",

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++ N I H +++++++++++++++++++++++++++++++++++++++++++++++++++
        # ---------------------------------------------- R E F   T R A I N ---------------------------------------------
        "train_contour_stream_nih":
            "images/nih/stream_images/contour/train",
        "train_lbp_stream_nih":
            "images/nih/stream_images/lbp/train",
        "train_rgb_stream_nih":
            "images/nih/stream_images/rgb/train",
        "train_texture_stream_nih":
            "images/nih/stream_images/texture/train",

        # ---------------------------------------------- R E F   V A L I D ---------------------------------------------
        "valid_contour_stream_nih":
            "images/nih/stream_images/contour/valid",
        "valid_lbp_stream_nih":
            "images/nih/stream_images/lbp/valid",
        "valid_rgb_stream_nih":
            "images/nih/stream_images/rgb/valid",
        "valid_texture_stream_nih":
            "images/nih/stream_images/texture/valid",

        # -------------------------------------------------- Q U E R Y -------------------------------------------------
        "test_contour_stream_ref_nih":
            "images/nih/test/ref/contour",
        "test_contour_stream_query_nih":
            "images/nih/test/query/contour",
        "test_lbp_stream_ref_nih":
            "images/nih/test/ref/lbp",
        "test_lbp_stream_query_nih":
            "images/nih/test/query/lbp",
        "test_rgb_stream_ref_nih":
            "images/nih/test/ref/rgb",
        "test_rgb_stream_query_nih":
            "images/nih/test/query/rgb",
        "test_texture_stream_ref_nih":
            "images/nih/test/ref/texture",
        "test_texture_stream_query_nih":
            "images/nih/test/query/texture",

        # ------------------------------------ P L O T T I N G   S T R E A M   N E T -----------------------------------
        "plotting_efficient_net_nih":
            "images/nih/plotting/stream_net/plotting_efficient_net",
        "plotting_efficient_net_v2_nih":
            "images/nih/plotting/stream_net/plotting_efficient_net_v2",

        # ------------------------------------ P L O T T I N G   F U S I O N   N E T -----------------------------------
        "plotting_fusion_network_efficient_net_self_attention_nih":
            "images/nih/plotting/fusion_net/plotting_fusion_network_efficient_net_self_attention",
        "plotting_fusion_network_efficient_net_v2_multi_head_attention_nih":
            "images/nih/plotting/fusion_net/plotting_fusion_network_efficient_net_v2_multi_head_attention",

        # ---------------------------- C O N F U S I O N   M A T R I X   S T R E A M   N E T ---------------------------
        "conf_mtx_efficient_net_nih":
            "images/nih/conf_mtx/stream_net/conf_mtx_efficient_net",
        "conf_mtx_efficient_net_v2_nih":
            "images/nih/conf_mtx/stream_net/conf_mtx_efficient_net_v2",

        # --------------------------- C O N F U S I O N   M A T R I X   F U S I O N   N E T ----------------------------
        "conf_mtx_fusion_network_efficient_net_self_attention_nih":
            "images/nih/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_self_attention",
        "conf_mtx_fusion_network_efficient_net_v2_multi_head_attention_nih":
            "images/nih/conf_mtx/fusion_net/conf_mtx_fusion_network_efficient_net_v2_multi_head_attention",

        # ++++++++++++++++++++++++++++++++++++++++++++++++++ O T H E R +++++++++++++++++++++++++++++++++++++++++++++++++
        # ----------------------------------------------- F O U R I E R ------------------------------------------------
        "Fourier_collected_images_by_shape_nih":
            "images/Fourier_desc/collected_images_by_shape_nih",
        "Fourier_euclidean_distance":
            "images/Fourier_desc/euclidean_distance",
        "Fourier_plot_shape":
            "images/Fourier_desc/plot_shape",
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
        # ------------------------------------------------- O T H E R --------------------------------------------------
        # k-fold directory names
        "Fourier_saved_mean_vectors":
            "data/Fourier_desc/saved_mean_vectors",

        "cure_k_fold":
            "data/cure/k_fold",
        "ogye_v2_k_fold":
            "data/ogyei/k_fold",
        "nih_k_fold":
            "data/nih/k_fold",

        # +++++++++++++++++++++++++++++++++++++++++++++++++++ C U R E ++++++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNet b0 - StreamNetwork
        "weights_efficient_net_contour_cure":
            "data/cure/weights/stream_net/efficient_net/weights_efficient_net_contour",
        "weights_efficient_net_lbp_cure":
            "data/cure/weights/stream_net/efficient_net/weights_efficient_net_lbp",
        "weights_efficient_net_rgb_cure":
            "data/cure/weights/stream_net/efficient_net/weights_efficient_net_rgb",
        "weights_efficient_net_texture_cure":
            "data/cure/weights/stream_net/efficient_net/weights_efficient_net_texture",

        # EfficientNet V2 - StreamNetwork
        "weights_efficient_net_v2_contour_cure":
            "data/cure/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_contour",
        "weights_efficient_net_v2_lbp_cure":
            "data/cure/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_lbp",
        "weights_efficient_net_v2_rgb_cure":
            "data/cure/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_rgb",
        "weights_efficient_net_v2_texture_cure":
            "data/cure/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_self_attention_cure":
            "data/cure/weights/fusion_net/efficient_net_self_attention",
        "weights_fusion_network_efficient_net_v2_multi_head_attention_cure":
            "data/cure/weights/fusion_net/efficient_net_v2_multi_head_attention",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNet b0
        "logs_efficient_net_contour_cure":
            "data/cure/logs/stream_net/efficient_net/logs_efficient_net_contour",
        "logs_efficient_net_lbp_cure":
            "data/cure/logs/stream_net/efficient_net/logs_efficient_net_lbp",
        "logs_efficient_net_rgb_cure":
            "data/cure/logs/stream_net/efficient_net/logs_efficient_net_rgb",
        "logs_efficient_net_texture_cure":
            "data/cure/logs/stream_net/efficient_net/logs_efficient_net_texture",

        # EfficientNet V2
        "logs_efficient_net_v2_contour_cure":
            "data/cure/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_contour",
        "logs_efficient_net_v2_lbp_cure":
            "data/cure/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_lbp",
        "logs_efficient_net_v2_rgb_cure":
            "data/cure/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_rgb",
        "logs_efficient_net_v2_texture_cure":
            "data/cure/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_self_attention_cure":
            "data/cure/logs/fusion_net/efficient_net_self_attention",
        "logs_fusion_network_efficient_net_v2_multi_head_attention_cure":
            "data/cure/logs/fusion_net/efficient_net_v2_multi_head",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_cure":
            "data/cure/predictions/stream_net/predictions_efficient_net",
        "predictions_efficient_net_v2_cure":
            "data/cure/predictions/stream_net/predictions_efficient_net_v2",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_self_attention_cure":
            "data/cure/predictions/fusion_net/predictions_fusion_network_efficient_net_self_attention_net",
        "predictions_fusion_network_efficient_net_v2_multi_head_attention_cure":
            "data/cure/predictions/fusion_net/predictions_fusion_network_efficient_net_v2_multi_head_attention",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_cure":
            "data/cure/ref_vec/stream_net/reference_vectors_efficient_net",
        "reference_vectors_efficient_net_v2_cure":
            "data/cure/ref_vec/stream_net/reference_vectors_efficient_net_v2",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "reference_vectors_fusion_network_efficient_net_self_attention_cure":
            "data/cure/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_self_attention",
        "reference_vectors_fusion_network_efficient_net_v2_multi_head_attention_cure":
            "data/cure/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_v2_multi_head_attention",


        # ++++++++++++++++++++++++++++++++++++++++++++++++++ O G Y E I +++++++++++++++++++++++++++++++++++++++++++++++++
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNet b0 - StreamNetwork
        "weights_efficient_net_contour_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_contour",
        "weights_efficient_net_lbp_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_lbp",
        "weights_efficient_net_rgb_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_rgb",
        "weights_efficient_net_texture_ogyei":
            "data/ogyei/weights/stream_net/efficient_net/weights_efficient_net_texture",

        # EfficientNet V2 - StreamNetwork
        "weights_efficient_net_v2_contour_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_contour",
        "weights_efficient_net_v2_lbp_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_lbp",
        "weights_efficient_net_v2_rgb_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_rgb",
        "weights_efficient_net_v2_texture_ogyei":
            "data/ogyei/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_self_attention_ogyei":
            "data/ogyei/weights/fusion_net/efficient_net_self_attention",
        "weights_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "data/ogyei/weights/fusion_net/efficient_net_v2_multi_head_attention",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNet b0
        "logs_efficient_net_contour_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_contour",
        "logs_efficient_net_lbp_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_lbp",
        "logs_efficient_net_rgb_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_rgb",
        "logs_efficient_net_texture_ogyei":
            "data/ogyei/logs/stream_net/efficient_net/logs_efficient_net_texture",

        # EfficientNet V2
        "logs_efficient_net_v2_contour_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_contour",
        "logs_efficient_net_v2_lbp_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_lbp",
        "logs_efficient_net_v2_rgb_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_rgb",
        "logs_efficient_net_v2_texture_ogyei":
            "data/ogyei/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_self_attention_ogyei":
            "data/ogyei/logs/fusion_net/efficient_net_self_attention",
        "logs_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "data/ogyei/logs/fusion_net/efficient_net_v2_multi_head",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_ogyei":
            "data/ogyei/predictions/stream_net/predictions_efficient_net",
        "predictions_efficient_net_v2_ogyei":
            "data/ogyei/predictions/stream_net/predictions_efficient_net_v2",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_self_attention_ogyei":
            "data/ogyei/predictions/fusion_net/predictions_fusion_network_efficient_net_self_attention_net",
        "predictions_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "data/ogyei/predictions/fusion_net/predictions_fusion_network_efficient_net_v2_multi_head_attention",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_ogyei":
            "data/ogyei/ref_vec/stream_net/reference_vectors_efficient_net",
        "reference_vectors_efficient_net_v2_ogyei":
            "data/ogyei/ref_vec/stream_net/reference_vectors_efficient_net_v2",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "reference_vectors_fusion_network_efficient_net_self_attention_ogyei":
            "data/ogyei/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_self_attention",
        "reference_vectors_fusion_network_efficient_net_v2_multi_head_attention_ogyei":
            "data/ogyei/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_v2_multi_head_attention",

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++ N I H +++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # ------------------------------------- W E I G H T S   S T R E A M   N E T ------------------------------------
        # EfficientNet b0 - StreamNetwork
        "weights_efficient_net_contour_nih":
            "data/nih/weights/stream_net/efficient_net/weights_efficient_net_contour",
        "weights_efficient_net_lbp_nih":
            "data/nih/weights/stream_net/efficient_net/weights_efficient_net_lbp",
        "weights_efficient_net_rgb_nih":
            "data/nih/weights/stream_net/efficient_net/weights_efficient_net_rgb",
        "weights_efficient_net_texture_nih":
            "data/nih/weights/stream_net/efficient_net/weights_efficient_net_texture",

        # EfficientNet V2 - StreamNetwork
        "weights_efficient_net_v2_contour_nih":
            "data/nih/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_contour",
        "weights_efficient_net_v2_lbp_nih":
            "data/nih/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_lbp",
        "weights_efficient_net_v2_rgb_nih":
            "data/nih/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_rgb",
        "weights_efficient_net_v2_texture_nih":
            "data/nih/weights/stream_net/efficient_net_v2/weights_efficient_v2_net_texture",

        # ------------------------------------- W E I G H T S   F U S I O N   N E T ------------------------------------
        "weights_fusion_network_efficient_net_self_attention_nih":
            "data/nih/weights/fusion_net/efficient_net_self_attention",
        "weights_fusion_network_efficient_net_v2_multi_head_attention_nih":
            "data/nih/weights/fusion_net/efficient_net_v2_multi_head_attention",

        # --------------------------------------- L O G S   S T R E A M   N E T ----------------------------------------
        # EfficientNet b0
        "logs_efficient_net_contour_nih":
            "data/nih/logs/stream_net/efficient_net/logs_efficient_net_contour",
        "logs_efficient_net_lbp_nih":
            "data/nih/logs/stream_net/efficient_net/logs_efficient_net_lbp",
        "logs_efficient_net_rgb_nih":
            "data/nih/logs/stream_net/efficient_net/logs_efficient_net_rgb",
        "logs_efficient_net_texture_nih":
            "data/nih/logs/stream_net/efficient_net/logs_efficient_net_texture",

        # EfficientNet V2
        "logs_efficient_net_v2_contour_nih":
            "data/nih/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_contour",
        "logs_efficient_net_v2_lbp_nih":
            "data/nih/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_lbp",
        "logs_efficient_net_v2_rgb_nih":
            "data/nih/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_rgb",
        "logs_efficient_net_v2_texture_nih":
            "data/nih/logs/stream_net/efficient_net_v2/logs_efficient_net_v2_texture",

        # ---------------------------------------- L O G S   F U S I O N   N E T ---------------------------------------
        "logs_fusion_network_efficient_net_self_attention_nih":
            "data/nih/logs/fusion_net/efficient_net_self_attention",
        "logs_fusion_network_efficient_net_v2_multi_head_attention_nih":
            "data/nih/logs/fusion_net/efficient_net_v2_multi_head",

        # -------------------------------- P R E D I C T I O N S    S T R E A M   N E T --------------------------------
        # Predictions
        "predictions_efficient_net_nih":
            "data/nih/predictions/stream_net/predictions_efficient_net",
        "predictions_efficient_net_v2_nih":
            "data/nih/predictions/stream_net/predictions_efficient_net_v2",

        # -------------------------------- P R E D I C T I O N S    F U S I O N   N E T --------------------------------
        # Predictions
        "predictions_fusion_network_efficient_net_self_attention_nih":
            "data/nih/predictions/fusion_net/predictions_fusion_network_efficient_net_self_attention_net",
        "predictions_fusion_network_efficient_net_v2_multi_head_attention_nih":
            "data/nih/predictions/fusion_net/predictions_fusion_network_efficient_net_v2_multi_head_attention",

        # -------------------------------------------- R E F   V E C T O R S -------------------------------------------
        "reference_vectors_efficient_net_nih":
            "data/nih/ref_vec/stream_net/reference_vectors_efficient_net",
        "reference_vectors_efficient_net_v2_nih":
            "data/nih/ref_vec/stream_net/reference_vectors_efficient_net_v2",

        # --------------------------------- R E F   V E C T O R S   F U S I O N   N E T --------------------------------
        "reference_vectors_fusion_net_efficient_net_self_attention_nih":
            "data/nih/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_self_attention",
        "reference_vectors_fusion_net_efficient_net_v2_multi_head_attention_nih":
            "data/nih/ref_vec/fusion_net/reference_vectors_fusion_net_efficient_net_v2_multi_head_attention",
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

        # ------------------------------------------------- O G Y E I --------------------------------------------------
        # Unsplitted
        "ogyei_v2_unsplitted_images":
            "ogyei_v2/unsplitted/images",
        "ogyei_v2_unsplitted_labels":
            "ogyei_v2/unsplitted/labels",
        "ogyei_v2_unsplitted_gt_masks":
            "ogyei_v2/unsplitted/gt_masks",

        # Train images, labels and masks
        "ogyei_v2_splitted_train_images":
            "ogyei_v2/splitted/train/images",
        "ogyei_v2_splitted_train_labels":
            "ogyei_v2/splitted/train/labels",
        "ogyei_v2_splitted_gt_train_masks":
            "ogyei_v2/splitted/train/gt_train_masks",
        "ogyei_v2_splitted_train_bbox_pixel_labels":
            "ogyei_v2/splitted/train/bbox_labels",
        "ogyei_v2_splitted_train_yolo_labels":
            "ogyei_v2/splitted/train/yolo_labels",
        "ogyei_v2_splitted_train_aug_images":
            "ogyei_v2/splitted_aug/train_dir/images",
        "ogyei_v2_splitted_train_aug_yolo_labels":
            "ogyei_v2/splitted_aug/train_dir/yolo_labels",
        "ogyei_v2_splitted_train_aug_mask_images":
            "ogyei_v2/splitted_aug/train_dir/mask_images",

        # Validation images, labels and masks
        "ogyei_v2_splitted_valid_images":
            "ogyei_v2/splitted/valid/images",
        "ogyei_v2_splitted_valid_labels":
            "ogyei_v2/splitted/valid/labels",
        "ogyei_v2_splitted_gt_valid_masks":
            "ogyei_v2/splitted/valid/gt_valid_masks",
        "ogyei_v2_splitted_valid_bbox_pixel_labels":
            "ogyei_v2/splitted/valid/bbox_labels",
        "ogyei_v2_splitted_valid_yolo_labels":
            "ogyei_v2/splitted/valid/yolo_labels",
        "ogyei_v2_splitted_valid_aug_images":
            "ogyei_v2/splitted_aug/valid_dir/images",
        "ogyei_v2_splitted_valid_aug_yolo_labels":
            "ogyei_v2/splitted_aug/valid_dir/yolo_labels",
        "ogyei_v2_splitted_valid_aug_mask_images":
            "ogyei_v2/splitted_aug/valid_dir/mask_images",

        # Test images, labels and masks
        "ogyei_v2_splitted_test_images":
            "ogyei_v2/splitted/test/images",
        "ogyei_v2_splitted_test_labels":
            "ogyei_v2/splitted/test/labels",
        "ogyei_v2_splitted_gt_test_masks":
            "ogyei_v2/splitted/test/gt_test_masks",
        "ogyei_v2_splitted_test_bbox_pixel_labels":
            "ogyei_v2/splitted/test/bbox_labels",
        "ogyei_v2_splitted_test_yolo_labels":
            "ogyei_v2/splitted/test/yolo_labels",
        "ogyei_v2_splitted_test_aug_images":
            "ogyei_v2/splitted_aug/test_dir/images",
        "ogyei_v2_splitted_test_aug_yolo_labels":
            "ogyei_v2/splitted_aug/test_dir/yolo_labels",
        "ogyei_v2_splitted_test_aug_mask_images":
            "ogyei_v2/splitted_aug/test_dir/mask_images",

        # --------------------------------------------------- N I H ----------------------------------------------------
        # CUSTOMER
        "nih_customer_images":
            "nih/Customer/images",
        "nih_customer_csv":
            "nih/Customer/csv",
        "nih_customer_xlsx":
            "nih/Customer/xlsx",
        "nih_customer_txt":
            "nih/Customer/txt",

        # REFERENCE
        "nih_reference_images":
            "cure/Reference/images",
        "nih_reference_masks":
            "nih/Reference/masks",
        "nih_reference_labels":
            "nih/Reference/labels",
        "nih_reference_csv":
            "nih/Reference/csv",
        "nih_reference_xlsx":
            "nih/Reference/xlsx",
        "nih_reference_txt":
            "nih/Reference/txt",

        # OTHER


        # -------------------------------------------------- C U R E ---------------------------------------------------
        # CUSTOMER
        "cure_customer_images":
            "cure/Customer/images",
        "cure_customer_segmentation_labels":
            "cure/Customer/segmentation_labels",
        "cure_customer_pixel_bbox_labels":
            "cure/Customer/pixel_bbox_labels",

        # REFERENCE
        "cure_reference_images":
            "cure/Reference/images",
        "cure_reference_masks":
            "cure/Reference/masks",
        "cure_reference_labels":
            "cure/Reference/labels",

        # CUSTOMER SPLITTED
        "cure_train_bbox_pixel_labels":
            "cure/Customer_splitted/train_dir/bbox_labels",
        "cure_train_images":
            "cure/Customer_splitted/train_dir/images",
        "cure_train_mask_images":
            "cure/Customer_splitted/train_dir/mask_images",
        "cure_train_segmentation_labels":
            "cure/Customer_splitted/train_dir/segmentation_labels",
        "cure_train_yolo_labels":
            "cure/Customer_splitted/train_dir/yolo_labels",

        "cure_valid_bbox_pixel_labels":
            "cure/Customer_splitted/valid_dir/bbox_labels",
        "cure_valid_images":
            "cure/Customer_splitted/valid_dir/images",
        "cure_valid_mask_images":
            "cure/Customer_splitted/valid_dir/mask_images",
        "cure_valid_segmentation_labels":
            "cure/Customer_splitted/valid_dir/segmentation_labels",
        "cure_valid_yolo_labels":
            "cure/Customer_splitted/valid_dir/yolo_labels",

        "cure_test_bbox_pixel_labels":
            "cure/Customer_splitted/test_dir/bbox_labels",
        "cure_test_images":
            "cure/Customer_splitted/test_dir/images",
        "cure_test_mask_images":
            "cure/Customer_splitted/test_dir/mask_images",
        "cure_test_segmentation_labels":
            "cure/Customer_splitted/test_dir/segmentation_labels",
        "cure_test_yolo_labels":
            "cure/Customer_splitted/test_dir/yolo_labels",

        # CUSTOMER SPLITTED AUGMENTED
        "cure_train_aug_images":
            "cure/Customer_splitted_aug/train_dir/images",
        "cure_train_aug_yolo_labels":
            "cure/Customer_splitted_aug/train_dir/yolo_labels",
        "cure_train_aug_mask_images":
            "cure/Customer_splitted_aug/train_dir/mask_images",

        "cure_valid_aug_images":
            "cure/Customer_splitted_aug/valid_dir/images",
        "cure_valid_aug_yolo_labels":
            "cure/Customer_splitted_aug/valid_dir/yolo_labels",
        "cure_valid_aug_mask_images":
            "cure/Customer_splitted_aug/valid_dir/mask_images",

        "cure_test_aug_images":
            "cure/Customer_splitted_aug/test_dir/images",
        "cure_test_aug_yolo_labels":
            "cure/Customer_splitted_aug/test_dir/yolo_labels"
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