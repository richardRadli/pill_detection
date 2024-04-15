"""
File: config_selector.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 10, 2023

Description:
The program holds the different configurations for the substream network, stream network and fusion network.
"""

import logging

from typing import Dict

from config.const import DATA_PATH, DATASET_PATH, IMAGES_PATH


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------- S U B N E T W O R K   C O N F I G S   T R A I N I N G -------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def sub_stream_network_configs(cfg) -> Dict:
    """
    Returns the dictionary containing the configuration details for the four different subnetworks
    (Contour, LBP, RGB, Texture) used in the StreamNetwork phase.

    :return: A dictionary containing the configuration details for the four subnetworks.
    """

    network_config = {
        # ----------------------------------------------- C O N T O U R ------------------------------------------------
        "Contour": {
            "channels":
                [1, 32, 48, 64, 128, 192, 256],
            "embedded_dim":
                128,
            "train":
                {
                    "cure": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_contour_stream_cure_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("contour_stream_cure_pos_neg")
                    },
                    "ogyei": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_contour_stream_ogyei_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("contour_stream_ogyei_pos_neg")
                        },
                },
            "ref":
                {
                    "cure":
                        IMAGES_PATH.get_data_path("test_contour_stream_ref_cure"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_contour_stream_ref_ogyei"),
                },
            "query":
                {
                    "cure":
                        IMAGES_PATH.get_data_path("test_contour_stream_query_cure"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_contour_stream_query_ogyei")
                },
            "model_weights_dir": {
                "EfficientNet":
                    {
                        "cure":
                            DATA_PATH.get_data_path("weights_efficient_net_contour_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_contour_ogyei"),
                    },
                "EfficientNetV2":
                    {
                        "cure":
                            DATA_PATH.get_data_path("weights_efficient_net_v2_contour_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_v2_contour_ogyei"),
                    },
            },
            "logs_dir": {
                "EfficientNet":
                        {
                            "cure":
                                DATA_PATH.get_data_path("logs_efficient_net_contour_cure"),
                            "ogyei":
                                DATA_PATH.get_data_path("logs_efficient_net_contour_ogyei"),
                        },
                "EfficientNetV2":
                        {
                            "cure":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_contour_cure"),
                            "ogyei":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_contour_ogyei"),
                        },
            },
            "hardest_samples": {
                "EfficientNet":
                    {
                        "cure":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_ogyei")
                    },
                "EfficientNetV2":
                    {
                        "cure":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei")
                    }
            },
            "learning_rate": {
                "EfficientNet":
                    cfg.learning_rate_en_con,
                "EfficientNetV2":
                    cfg.learning_rate_env2_con
            }.get(cfg.type_of_net, cfg.learning_rate_en_con),
            "image_size": {
                "EfficientNet":
                    cfg.img_size_en,
                "EfficientNetV2":
                    cfg.img_size_en
            }.get(cfg.type_of_net, cfg.img_size_en),
            "grayscale": True
        },

        # --------------------------------------------------- L B P ----------------------------------------------------
        "LBP": {
            "channels":
                [1, 32, 48, 64, 128, 192, 256],
            "embedded_dim":
                128,
            "train":
                {
                    "cure": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_lbp_stream_cure_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("lbp_stream_cure_pos_neg")
                    },
                    "ogyei": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_lbp_stream_ogyei_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("lbp_stream_ogyei_pos_neg")
                        },
                },
            "ref":
                {
                    "cure":
                        IMAGES_PATH.get_data_path("test_lbp_stream_ref_cure"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_lbp_stream_ref_ogyei"),
                },
            "query":
                {
                    "cure":
                        IMAGES_PATH.get_data_path("test_lbp_stream_query_cure"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_lbp_stream_query_ogyei")
                },
            "model_weights_dir": {
                "EfficientNet":
                    {
                        "cure":
                            DATA_PATH.get_data_path("weights_efficient_net_lbp_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_lbp_ogyei"),
                    },
                "EfficientNetV2":
                    {
                        "cure":
                            DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_ogyei"),
                    },
            },
            "logs_dir": {
                "EfficientNet":
                        {
                            "cure":
                                DATA_PATH.get_data_path("logs_efficient_net_lbp_cure"),
                            "ogyei":
                                DATA_PATH.get_data_path("logs_efficient_net_lbp_ogyei"),
                        },
                "EfficientNetV2":
                        {
                            "cure":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_cure"),
                            "ogyei":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_ogyei"),
                        },
            },
            "hardest_samples": {
                "EfficientNet":
                    {
                        "cure":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_ogyei")
                    },
                "EfficientNetV2":
                    {
                        "cure":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei")
                    }
            },
            "learning_rate": {
                "EfficientNet":
                    cfg.learning_rate_en_con,
                "EfficientNetV2":
                    cfg.learning_rate_env2_con
            }.get(cfg.type_of_net, cfg.learning_rate_en_con),
            "image_size": {
                "EfficientNet":
                    cfg.img_size_en,
                "EfficientNetV2":
                    cfg.img_size_en
            }.get(cfg.type_of_net, cfg.img_size_en),
            "grayscale": True
        },
        # --------------------------------------------------- R G B ----------------------------------------------------
        "RGB": {
            "channels":
                [3, 32, 48, 64, 128, 192, 256],
            "embedded_dim":
                256,
            "train":
                {
                    "cure": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_rgb_stream_cure_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("rgb_stream_cure_pos_neg")
                    },
                    "ogyei": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_rgb_stream_ogyei_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("rgb_stream_ogyei_pos_neg")
                        },
                },
            "ref":
                {
                    "cure":
                        IMAGES_PATH.get_data_path("test_rgb_stream_ref_cure"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_rgb_stream_ref_ogyei"),
                },
            "query":
                {
                    "cure":
                        IMAGES_PATH.get_data_path("test_rgb_stream_query_cure"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_rgb_stream_query_ogyei")
                },
            "model_weights_dir": {
                "EfficientNet":
                    {
                        "cure":
                            DATA_PATH.get_data_path("weights_efficient_net_rgb_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_rgb_ogyei"),
                    },
                "EfficientNetV2":
                    {
                        "cure":
                            DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_ogyei"),
                    },
            },
            "logs_dir": {
                "EfficientNet":
                        {
                            "cure":
                                DATA_PATH.get_data_path("logs_efficient_net_rgb_cure"),
                            "ogyei":
                                DATA_PATH.get_data_path("logs_efficient_net_rgb_ogyei"),
                        },
                "EfficientNetV2":
                        {
                            "cure":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_cure"),
                            "ogyei":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_ogyei"),
                        },
            },
            "hardest_samples": {
                "EfficientNet":
                    {
                        "cure":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_ogyei")
                    },
                "EfficientNetV2":
                    {
                        "cure":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei")
                    }
            },
            "learning_rate": {
                "EfficientNet":
                    cfg.learning_rate_en_con,
                "EfficientNetV2":
                    cfg.learning_rate_env2_con
            }.get(cfg.type_of_net, cfg.learning_rate_en_con),
            "image_size": {
                "EfficientNet":
                    cfg.img_size_en,
                "EfficientNetV2":
                    cfg.img_size_en
            }.get(cfg.type_of_net, cfg.img_size_en),
            "grayscale": False
        },
        # ----------------------------------------------- T E X T U R E ------------------------------------------------
        "Texture": {
            "channels":
                [1, 32, 48, 64, 128, 192, 256],
            "embedded_dim":
                128,
            "train":
                {
                    "cure": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_texture_stream_cure_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("texture_stream_cure_pos_neg")
                    },
                    "ogyei": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_texture_stream_ogyei_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("texture_stream_ogyei_pos_neg")
                        },
                },
            "ref":
                {
                    "cure":
                        IMAGES_PATH.get_data_path("test_texture_stream_ref_cure"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_texture_stream_ref_ogyei"),
                },
            "query":
                {
                    "cure":
                        IMAGES_PATH.get_data_path("test_texture_stream_query_cure"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_texture_stream_query_ogyei")
                },
            "model_weights_dir": {
                "EfficientNet":
                    {
                        "cure":
                            DATA_PATH.get_data_path("weights_efficient_net_texture_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_texture_ogyei"),
                    },
                "EfficientNetV2":
                    {
                        "cure":
                            DATA_PATH.get_data_path("weights_efficient_net_v2_texture_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_v2_texture_ogyei"),
                    },
            },
            "logs_dir": {
                "EfficientNet":
                        {
                            "cure":
                                DATA_PATH.get_data_path("logs_efficient_net_texture_cure"),
                            "ogyei":
                                DATA_PATH.get_data_path("logs_efficient_net_texture_ogyei"),
                        },
                "EfficientNetV2":
                        {
                            "cure":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_texture_cure"),
                            "ogyei":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_texture_ogyei"),
                        },
            },
            "hardest_samples": {
                "EfficientNet":
                    {
                        "cure":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_ogyei")
                    },
                "EfficientNetV2":
                    {
                        "cure":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei")
                    }
            },
            "learning_rate": {
                "EfficientNet":
                    cfg.learning_rate_en_con,
                "EfficientNetV2":
                    cfg.learning_rate_env2_con
            }.get(cfg.type_of_net, cfg.learning_rate_en_con),
            "image_size": {
                "EfficientNet":
                    cfg.img_size_en,
                "EfficientNetV2":
                    cfg.img_size_en
            }.get(cfg.type_of_net, cfg.img_size_en),
            "grayscale": True
        }
    }

    return network_config


# ------------------------------------------------------------------------------------------------------------------
# ---------------------------------- G E T   M A I N   N E T W O R K   C O N F I G ---------------------------------
# ------------------------------------------------------------------------------------------------------------------
def stream_network_config(cfg) -> Dict:
    """
    Returns a dictionary containing the prediction, plotting, and reference vectors folder paths for different types of
    networks based on the type_of_net parameter in cfg.
    :return: Dictionary containing the prediction and plotting folder paths.
    """

    network_type = cfg.type_of_net
    logging.info(network_type)
    network_configs = {
        'EfficientNet': {
            'prediction_folder': {
                "cure":
                    DATA_PATH.get_data_path("predictions_efficient_net_cure"),
                "ogyei":
                    DATA_PATH.get_data_path("predictions_efficient_net_ogyei")
            },
            'plotting_folder': {
                "cure":
                    IMAGES_PATH.get_data_path("plotting_efficient_net_cure"),
                "ogyei":
                    IMAGES_PATH.get_data_path("plotting_efficient_net_ogyei")
            },
            'confusion_matrix': {
                "cure":
                    IMAGES_PATH.get_data_path("conf_mtx_efficient_net_cure"),
                "ogyei":
                    IMAGES_PATH.get_data_path("conf_mtx_efficient_net_ogyei")
            },
            'ref_vectors_folder': {
                "cure":
                    DATA_PATH.get_data_path("reference_vectors_efficient_net_cure"),
                "ogyei":
                    DATA_PATH.get_data_path("reference_vectors_efficient_net_ogyei"),
            },
            'hard_sample': {
                "Contour": {
                    "cure":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_ogyei")
                },
                "LBP": {
                    "cure":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_ogyei")
                },
                "RGB": {
                    "cure":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_ogyei")
                },
                "Texture": {
                    "cure":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_ogyei")
                }
            }
        },
        'EfficientNetV2': {
            'prediction_folder': {
                "cure":
                    DATA_PATH.get_data_path("predictions_efficient_net_v2_cure"),
                "ogyei":
                    DATA_PATH.get_data_path("predictions_efficient_net_v2_ogyei")
            },
            'plotting_folder': {
                "cure":
                    IMAGES_PATH.get_data_path("plotting_efficient_net_v2_cure"),
                "ogyei":
                    IMAGES_PATH.get_data_path("plotting_efficient_net_v2_ogyei")
            },
            'confusion_matrix': {
                "cure":
                    IMAGES_PATH.get_data_path("conf_mtx_efficient_net_v2_cure"),
                "ogyei":
                    IMAGES_PATH.get_data_path("conf_mtx_efficient_net_v2_ogyei")
            },
            'ref_vectors_folder': {
                "cure":
                    DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_cure"),
                "ogyei":
                    DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_ogyei"),
            },
            'hard_sample': {
                "Contour": {
                    "cure":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei")
                },
                "LBP": {
                    "cure":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei")
                },
                "RGB": {
                    "cure":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei")
                },
                "Texture": {
                    "cure":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei")
                }
            }
        },
    }
    if network_type not in network_configs:
        raise ValueError(f'Invalid network type: {network_type}')

    return network_configs[network_type]


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------ M A I N   N E T W O R K   C O N F I G   F U S I O N   T R A I N I N G -----------------------
# ----------------------------------------------------------------------------------------------------------------------
def fusion_network_config(network_type) -> Dict:
    """
    Returns a dictionary containing the prediction, plotting, and reference vectors folder paths for different types of
    networks based on the type_of_net parameter in cfg.
    :return: Dictionary containing the prediction and plotting folder paths.
    """

    network_configs = {
        'EfficientNetSelfAttention': {
            'logs_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path("logs_fusion_network_efficient_net_self_attention_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("logs_fusion_network_efficient_net_self_attention_ogyei"),
                },
            'weights_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path("weights_fusion_network_efficient_net_self_attention_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("weights_fusion_network_efficient_net_self_attention_ogyei"),
                },
            'prediction_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path("predictions_fusion_network_efficient_net_self_attention_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("predictions_fusion_network_efficient_net_self_attention_ogyei"),
                },
            'plotting_folder':
                {
                    "cure":
                        IMAGES_PATH.get_data_path("plotting_fusion_network_efficient_net_self_attention_cure"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("plotting_fusion_network_efficient_net_self_attention_ogyei"),
                },
            'confusion_matrix':
                {
                    "cure":
                        IMAGES_PATH.get_data_path("conf_mtx_fusion_network_efficient_net_self_attention_cure"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("conf_mtx_fusion_network_efficient_net_self_attention_ogyei"),
                },
            'ref_vectors_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path("reference_vectors_fusion_network_efficient_net_self_attention_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("reference_vectors_fusion_network_efficient_net_self_attention_ogyei"),
                },
        },
        "EfficientNetV2MultiHeadAttention": {
            'logs_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path("logs_fusion_network_efficient_net_v2_multi_head_attention_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("logs_fusion_network_efficient_net_v2_multi_head_attention_ogyei"),
                },
            'weights_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path("weights_fusion_network_efficient_net_v2_multi_head_attention_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("weights_fusion_network_efficient_net_v2_multi_head_attention_ogyei"),
                },
            'prediction_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_v2_multi_head_attention_cure"
                        ),
                    "ogyei":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_v2_multi_head_attention_ogyei"
                        ),
                },
            'plotting_folder':
                {
                    "cure":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_v2_multi_head_attention_cure"
                        ),
                    "ogyei":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_v2_multi_head_attention_ogyei"
                        ),
                },
            'confusion_matrix':
                {
                    "cure":
                        IMAGES_PATH.get_data_path(
                            "conf_mtx_fusion_network_efficient_net_v2_multi_head_attention_cure"
                        ),
                    "ogyei":
                        IMAGES_PATH.get_data_path(
                            "conf_mtx_fusion_network_efficient_net_v2_multi_head_attention_ogyei"
                        ),
                },
            'ref_vectors_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path(
                            "reference_vectors_fusion_network_efficient_net_v2_multi_head_attention_cure"
                        ),
                    "ogyei":
                        DATA_PATH.get_data_path(
                            "reference_vectors_fusion_network_efficient_net_v2_multi_head_attention_ogyei"
                        ),
                },
        },
        "EfficientNetV2MHAFMHA": {
            'logs_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path("logs_fusion_network_efficient_net_v2_mha_fmha_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("logs_fusion_network_efficient_net_v2_mha_fmha_ogyei"),
                },
            'weights_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path("weights_fusion_network_efficient_net_v2_mha_fmha_cure"),
                    "ogyei":
                        DATA_PATH.get_data_path("weights_fusion_network_efficient_net_v2_mha_fmha_ogyei"),
                },
            'prediction_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_v2_mha_fmha_cure"
                        ),
                    "ogyei":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_v2_mha_fmha_ogyei"
                        ),
                },
            'plotting_folder':
                {
                    "cure":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_v2_mha_fmha_cure"
                        ),
                    "ogyei":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_v2_mha_fmha_ogyei"
                        ),
                },
            'confusion_matrix':
                {
                    "cure":
                        IMAGES_PATH.get_data_path(
                            "conf_mtx_fusion_network_efficient_net_v2_mha_fmha_cure"
                        ),
                    "ogyei":
                        IMAGES_PATH.get_data_path(
                            "conf_mtx_fusion_network_efficient_net_v2_mha_fmha_ogyei"
                        ),
                },
            'ref_vectors_folder':
                {
                    "cure":
                        DATA_PATH.get_data_path(
                            "reference_vectors_fusion_network_efficient_net_v2_mha_fmha_cure"
                        ),
                    "ogyei":
                        DATA_PATH.get_data_path(
                            "reference_vectors_fusion_network_efficient_net_v2_mha_fmha_ogyei"
                        ),
                },
        },
    }
    if network_type not in network_configs:
        raise ValueError(f'Invalid network type: {network_type}')

    return network_configs[network_type]


def dataset_images_path_selector(dataset_name):
    """
    Selects the correct directory paths based on the given operation string.

    :return: A dictionary containing directory paths for images, masks, and other related files.
    :raises ValueError: If the operation string is not "train" or "test".
    """

    path_to_images = {
        # --------------------------------------------------- C U R E --------------------------------------------------
        "cure": {
            "customer": {
                "customer_images":
                    DATASET_PATH.get_data_path("cure_customer_images"),
                "customer_segmentation_labels":
                    DATASET_PATH.get_data_path("cure_customer_segmentation_labels"),
                "customer_mask_images":
                    DATASET_PATH.get_data_path("cure_customer_mask_images")
            },

            "reference": {
                "reference_images":
                    DATASET_PATH.get_data_path("cure_reference_images"),
                "reference_mask_images":
                    DATASET_PATH.get_data_path("cure_reference_mask_images"),
                "reference_labels":
                    DATASET_PATH.get_data_path("cure_reference_yolo_labels"),
            },

            "src_stream_images": {
                "reference": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_reference"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_reference_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_reference_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_reference_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_reference_texture"),
                },
                "customer": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_customer"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_customer_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_customer_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_customer_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_customer_texture"),
                }
            },

            "dst_stream_images": {
                'stream_images_anchor':
                    IMAGES_PATH.get_data_path("stream_images_cure_anchor"),
                "stream_images_pos_neg":
                    IMAGES_PATH.get_data_path("stream_images_cure_pos_neg"),
                'ref':
                    IMAGES_PATH.get_data_path("test_ref_cure"),
                'query':
                    IMAGES_PATH.get_data_path("test_query_cure")
            },

            "other": {
                'k_fold':
                    DATA_PATH.get_data_path("cure_k_fold")
            },
        },
        # -------------------------------------------------- O G Y E I -------------------------------------------------
        "ogyei": {
            "customer": {
                "customer_images":
                    DATASET_PATH.get_data_path("ogyei_customer_images"),
                "customer_mask_images":
                    DATASET_PATH.get_data_path("ogyei_customer_mask_images"),
                "customer_segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_customer_segmentation_labels")
            },

            "reference": {
                "reference_images":
                    DATASET_PATH.get_data_path("ogyei_reference_images"),
                "reference_mask_images":
                    DATASET_PATH.get_data_path("ogyei_reference_mask_images"),
                "reference_segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_reference_segmentation_labels")
            },

            "src_stream_images": {
                "reference": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_ogyei_reference"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_ogyei_reference_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_ogyei_reference_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_ogyei_reference_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_ogyei_reference_texture"),
                },
                "customer": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_ogyei_customer"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_ogyei_customer_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_ogyei_customer_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_ogyei_customer_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_ogyei_customer_texture"),
                }
            },

            "dst_stream_images": {
                'stream_images_anchor':
                    IMAGES_PATH.get_data_path("stream_images_ogyei_anchor"),
                "stream_images_pos_neg":
                    IMAGES_PATH.get_data_path("stream_images_ogyei_pos_neg"),
                'ref':
                    IMAGES_PATH.get_data_path("test_ref_ogyei"),
                'query':
                    IMAGES_PATH.get_data_path("test_query_ogyei")
            }
        },

        # ---------------------------------------------------- D T D ---------------------------------------------------
        "dtd": {
            "dataset_path": DATASET_PATH.get_data_path("dtd_images")
        }
    }

    return path_to_images[dataset_name]
