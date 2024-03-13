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
# ----------------------------------------- C A M E R A   C O N F I G --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def word_embedded_network_configs(dataset_name=None) -> Dict:
    word_embedded_network_config_config = {
        "cure_one_sided": {
            "input_dim":
                70,
            "hidden_dim":
                100,
            "output_dim":
                3,
            "model_weights_dir":
                DATA_PATH.get_data_path("weights_word_embedded_network_cure_one_sided"),
            "logs_dir":
                DATA_PATH.get_data_path("logs_word_embedded_network_cure_one_sided"),
            "predictions":
                DATA_PATH.get_data_path("predictions_word_embedded_network_cure_one_sided")
        },
        "cure_two_sided": {
            "input_dim":
                70,
            "hidden_dim":
                100,
            "output_dim":
                3,
            "model_weights_dir":
                DATA_PATH.get_data_path("weights_word_embedded_network_cure_two_sided"),
            "logs_dir":
                DATA_PATH.get_data_path("logs_word_embedded_network_cure_two_sided"),
            "predictions":
                DATA_PATH.get_data_path("predictions_word_embedded_network_cure_two_sided")
        },
        "ogyei": {
            "input_dim":
                70,
            "hidden_dim":
                100,
            "output_dim":
                3,
            "model_weights_dir":
                DATA_PATH.get_data_path("weights_word_embedded_network_ogyei"),
            "logs_dir":
                DATA_PATH.get_data_path("logs_word_embedded_network_ogyei"),
            "predictions":
                DATA_PATH.get_data_path("predictions_word_embedded_ogyei")
        }
    }

    return word_embedded_network_config_config[dataset_name]


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
                    "cure_two_sided": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_contour_stream_cure_two_sided_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("contour_stream_cure_two_sided_pos_neg")
                    },
                    "cure_one_sided": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_contour_stream_cure_one_sided_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("contour_stream_cure_one_sided_pos_neg")
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
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path("test_contour_stream_ref_cure_two_sided"),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path("test_contour_stream_ref_cure_one_sided"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_contour_stream_ref_ogyei"),
                },
            "query":
                {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path("test_contour_stream_query_cure_two_sided"),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path("test_contour_stream_query_cure_one_sided"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_contour_stream_query_ogyei")
                },
            "model_weights_dir": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("weights_efficient_net_contour_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("weights_efficient_net_contour_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_contour_ogyei"),
                    },
            },
            "logs_dir": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("logs_efficient_net_contour_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("logs_efficient_net_contour_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("logs_efficient_net_contour_ogyei"),
                    },
            },
            "hardest_samples": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_ogyei")
                    }
            },
            "learning_rate": {
                "EfficientNet":
                    cfg.learning_rate_en_con,
            }.get(cfg.type_of_net, cfg.learning_rate_en_con),
            "image_size": {
                "EfficientNet":
                    cfg.img_size_en,
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
                    "cure_two_sided": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_lbp_stream_cure_two_sided_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("lbp_stream_cure_two_sided_pos_neg")
                    },
                    "cure_one_sided": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_lbp_stream_cure_one_sided_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("lbp_stream_cure_one_sided_pos_neg")
                    },
                    "ogyei": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_lbp_stream_ogyei_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("lbp_stream_ogyei_pos_neg")
                    }

                },
            "ref":
                {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path("test_lbp_stream_ref_cure_two_sided"),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path("test_lbp_stream_ref_cure_one_sided"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_lbp_stream_ref_ogyei"),
                },
            "query":
                {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path("test_lbp_stream_query_cure_two_sided"),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path("test_lbp_stream_query_cure_one_sided"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_lbp_stream_query_ogyei")
                },
            "model_weights_dir": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("weights_efficient_net_lbp_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("weights_efficient_net_lbp_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_lbp_ogyei"),
                    },
            },
            "logs_dir": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("logs_efficient_net_lbp_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("logs_efficient_net_lbp_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("logs_efficient_net_lbp_ogyei"),
                    }
            },
            "hardest_samples": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_ogyei")
                    }
            },
            "learning_rate": {
                "EfficientNet":
                    cfg.learning_rate_en_lbp,
            }.get(cfg.type_of_net, cfg.learning_rate_en_lbp),
            "image_size": {
                "EfficientNet":
                    cfg.img_size_en,
            }.get(cfg.type_of_net, cfg.img_size_en),
            "grayscale": True
        },
        # --------------------------------------------------- R G B ----------------------------------------------------
        "RGB": {
            "channels":
                [3, 64, 96, 128, 256, 384, 512],
            "embedded_dim":
                256,
            "train":
                {
                    "cure_two_sided": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_rgb_stream_cure_two_sided_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("rgb_stream_cure_two_sided_pos_neg")
                    },
                    "cure_one_sided": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_rgb_stream_cure_one_sided_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("rgb_stream_cure_one_sided_pos_neg")
                    },
                    "ogyei": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_rgb_stream_ogyei_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("rgb_stream_ogyei_pos_neg")
                    }
                },
            "ref":
                {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path("test_rgb_stream_ref_cure_two_sided"),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path("test_rgb_stream_ref_cure_one_sided"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_rgb_stream_ref_ogyei"),
                },
            "query":
                {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path("test_rgb_stream_query_cure_two_sided"),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path("test_rgb_stream_query_cure_one_sided"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_rgb_stream_query_ogyei")
                },
            "model_weights_dir": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("weights_efficient_net_rgb_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("weights_efficient_net_rgb_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_rgb_ogyei"),
                    },
            },
            "logs_dir": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("logs_efficient_net_rgb_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("logs_efficient_net_rgb_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("logs_efficient_net_rgb_ogyei"),
                    },
            },
            "hardest_samples": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_ogyei")
                    }
            },
            "learning_rate": {
                "EfficientNet":
                    cfg.learning_rate_en_rgb,
            }.get(cfg.type_of_net, cfg.learning_rate_en_rgb),
            "image_size": {
                "EfficientNet":
                    cfg.img_size_en,
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
                    "cure_two_sided": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_texture_stream_cure_two_sided_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("texture_stream_cure_two_sided_pos_neg")
                    },
                    "cure_one_sided": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_texture_stream_cure_one_sided_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("texture_stream_cure_one_sided_pos_neg")
                    },
                    "ogyei": {
                        "anchor":
                            IMAGES_PATH.get_data_path("train_texture_stream_ogyei_anchor"),
                        "pos_neg":
                            IMAGES_PATH.get_data_path("texture_stream_ogyei_pos_neg")
                    }
                },
            "ref":
                {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path("test_texture_stream_ref_cure_two_sided"),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path("test_texture_stream_ref_cure_one_sided"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_texture_stream_ref_ogyei"),
                },
            "query":
                {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path("test_texture_stream_query_cure_two_sided"),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path("test_texture_stream_query_cure_one_sided"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("test_texture_stream_query_ogyei")
                },
            "model_weights_dir": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("weights_efficient_net_texture_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("weights_efficient_net_texture_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("weights_efficient_net_texture_ogyei"),
                    }
            },
            "logs_dir": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("logs_efficient_net_texture_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("logs_efficient_net_texture_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("logs_efficient_net_texture_ogyei"),
                    }
            },
            "hardest_samples": {
                "EfficientNet":
                    {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_ogyei")
                    }
            },
            "learning_rate": {
                "EfficientNet":
                    cfg.learning_rate_en_tex,
            }.get(cfg.type_of_net, cfg.learning_rate_en_tex),
            "image_size": {
                "EfficientNet":
                    cfg.img_size_en,
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
        'EfficientNet':
            {
                'prediction_folder': {
                    "cure_two_sided":
                        DATA_PATH.get_data_path("predictions_efficient_net_cure_two_sided"),
                    "cure_one_sided":
                        DATA_PATH.get_data_path("predictions_efficient_net_cure_one_sided"),
                    "ogyei":
                        DATA_PATH.get_data_path("predictions_efficient_net_ogyei")
                },
                'plotting_folder': {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path("plotting_efficient_net_cure_two_sided"),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path("plotting_efficient_net_cure_one_sided"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("plotting_efficient_net_ogyei")
                },
                'confusion_matrix': {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path("conf_mtx_efficient_net_cure_two_sided"),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path("conf_mtx_efficient_net_cure_one_sided"),
                    "ogyei":
                        IMAGES_PATH.get_data_path("conf_mtx_efficient_net_ogyei")
                },
                'ref_vectors_folder': {
                    "cure_two_sided":
                        DATA_PATH.get_data_path("reference_vectors_efficient_net_cure_two_sided"),
                    "cure_one_sided":
                        DATA_PATH.get_data_path("reference_vectors_efficient_net_cure_one_sided"),
                    "ogyei":
                        DATA_PATH.get_data_path("reference_vectors_efficient_net_ogyei")
                },
                'hard_sample': {
                    "Contour": {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_ogyei")
                    },
                    "LBP": {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_ogyei")
                    },
                    "RGB": {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_ogyei")
                    },
                    "Texture": {
                        "cure_two_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_cure_two_sided"),
                        "cure_one_sided":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_cure_one_sided"),
                        "ogyei":
                            DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_ogyei")
                    }
                }
            }
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
        'EfficientNetMultiHeadAttention': {
            'logs_folder':
                {
                    "cure_two_sided":
                        DATA_PATH.get_data_path(
                            "logs_fusion_network_efficient_net_multi_head_attention_cure_two_sided"),
                    "cure_one_sided":
                        DATA_PATH.get_data_path(
                            "logs_fusion_network_efficient_net_multi_head_attention_cure_one_sided"),
                    "ogyei":
                        DATA_PATH.get_data_path("logs_fusion_network_efficient_net_multi_head_attention_ogyei"),
                },
            'weights_folder':
                {
                    "cure_two_sided":
                        DATA_PATH.get_data_path(
                            "weights_fusion_network_efficient_net_multi_head_attention_cure_two_sided"),
                    "cure_one_sided":
                        DATA_PATH.get_data_path(
                            "weights_fusion_network_efficient_net_multi_head_attention_cure_one_sided"),
                    "ogyei":
                        DATA_PATH.get_data_path("weights_fusion_network_efficient_net_multi_head_attention_ogyei"),
                },
            'prediction_folder':
                {
                    "cure_two_sided":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_multi_head_attention_cure_two_sided"
                        ),
                    "cure_one_sided":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_multi_head_attention_cure_one_sided"
                        ),
                    "ogyei":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_multi_head_attention_ogyei"
                        ),
                },
            'plotting_folder':
                {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_multi_head_attention_cure_two_sided"
                        ),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_multi_head_attention_cure_one_sided"
                        ),
                    "ogyei":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_multi_head_attention_ogyei"
                        ),
                },
            'confusion_matrix':
                {
                    "cure_two_sided":
                        IMAGES_PATH.get_data_path(
                            "conf_mtx_fusion_network_efficient_net_multi_head_attention_cure_two_sided"
                        ),
                    "cure_one_sided":
                        IMAGES_PATH.get_data_path(
                            "conf_mtx_fusion_network_efficient_net_multi_head_attention_cure_one_sided"
                        ),
                    "ogyei":
                        IMAGES_PATH.get_data_path(
                            "conf_mtx_fusion_network_efficient_net_multi_head_attention_ogyei"
                        ),
                },
            'ref_vectors_folder':
                {
                    "cure_two_sided":
                        DATA_PATH.get_data_path(
                            "reference_vectors_fusion_network_efficient_net_multi_head_attention_cure_two_sided"
                        ),
                    "cure_one_sided":
                        DATA_PATH.get_data_path(
                            "reference_vectors_fusion_network_efficient_net_multi_head_attention_cure_one_sided"
                        ),
                    "ogyei":
                        DATA_PATH.get_data_path(
                            "reference_vectors_fusion_network_efficient_net_multi_head_attention_ogyei"
                        ),
                }
        }
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
        "cure_two_sided": {
            "customer": {
                "customer_images":
                    DATASET_PATH.get_data_path("cure_two_sided_customer_images"),
                "customer_segmentation_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_customer_segmentation_labels"),
                "customer_mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_customer_mask_images")
            },

            "reference": {
                "reference_images":
                    DATASET_PATH.get_data_path("cure_two_sided_reference_images"),
                "reference_mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_reference_mask_images"),
                "reference_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_reference_yolo_labels"),
            },

            "train": {
                "images":
                    DATASET_PATH.get_data_path("cure_two_sided_train_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_train_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_train_segmentation_labels"),
                "aug_images":
                    DATASET_PATH.get_data_path("cure_two_sided_train_aug_images"),
                "aug_yolo_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_train_aug_yolo_labels"),
                "aug_mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_train_aug_mask_images"),
            },

            "valid": {
                "images":
                    DATASET_PATH.get_data_path("cure_two_sided_valid_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_valid_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_valid_segmentation_labels"),
                "aug_images":
                    DATASET_PATH.get_data_path("cure_two_sided_valid_aug_images"),
                "aug_yolo_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_valid_aug_yolo_labels"),
                "aug_mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_valid_aug_mask_images"),
            },

            "test": {
                "images":
                    DATASET_PATH.get_data_path("cure_two_sided_test_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_two_sided_test_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_test_segmentation_labels"),
                "aug_images":
                    DATASET_PATH.get_data_path("cure_two_sided_test_aug_images"),
                "aug_yolo_labels":
                    DATASET_PATH.get_data_path("cure_two_sided_test_aug_yolo_labels"),
            },

            "src_stream_images": {
                "reference": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_reference_texture"),
                },
                "customer": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_customer"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_customer_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_customer_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_customer_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_two_sided_customer_texture"),
                }
            },

            "dst_stream_images": {
                'stream_images_anchor':
                    IMAGES_PATH.get_data_path("stream_images_cure_two_sided_anchor"),
                "stream_images_pos_neg":
                    IMAGES_PATH.get_data_path("stream_images_cure_two_sided_pos_neg"),
                'ref':
                    IMAGES_PATH.get_data_path("test_ref_cure_two_sided"),
                'query':
                    IMAGES_PATH.get_data_path("test_query_cure_two_sided")
            },

            "other": {
                'k_fold':
                    DATA_PATH.get_data_path("cure_two_sided_k_fold")
            },

            "dynamic_margin": {
                "pill_desc_xlsx":
                    DATA_PATH.get_data_path("pill_desc_xlsx_cure_two_sided"),
                "Fourier_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_cure_two_sided"),
                "Fourier_euclidean_distance":
                    IMAGES_PATH.get_data_path("Fourier_euclidean_distance_cure_two_sided"),
                "Fourier_saved_mean_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_cure_two_sided"),
                "Fourier_images_by_shape":
                    IMAGES_PATH.get_data_path("Fourier_collected_images_by_shape_cure_two_sided"),
                "colour_vectors":
                    DATA_PATH.get_data_path("colour_vectors_cure_two_sided"),
                "imprint_vectors":
                    DATA_PATH.get_data_path("imprint_vectors_cure_two_sided"),
                "score_vectors":
                    DATA_PATH.get_data_path("score_vectors_cure_two_sided"),
                "concatenated_vectors":
                    DATA_PATH.get_data_path("concatenated_vectors_cure_two_sided")
            }
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

            "train": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_train_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_train_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_train_segmentation_labels")
            },

            "valid": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_valid_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_valid_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_valid_segmentation_labels")
            },

            "test": {
                "images":
                    DATASET_PATH.get_data_path("ogyei_test_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("ogyei_test_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("ogyei_test_segmentation_labels"),
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
            },

            "other": {
                "ref": DATASET_PATH.get_data_path("ogyei_ref_images"),
                "query": DATASET_PATH.get_data_path("ogyei_query_images"),
                'k_fold': DATA_PATH.get_data_path("ogyei_k_fold")
            },

            "dynamic_margin": {
                "pill_desc_xlsx":
                    DATA_PATH.get_data_path("pill_desc_xlsx_ogyei"),
                "Fourier_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_ogyei"),
                "Fourier_euclidean_distance":
                    IMAGES_PATH.get_data_path("Fourier_euclidean_distance_ogyei"),
                "Fourier_saved_mean_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_ogyei"),
                "Fourier_images_by_shape":
                    IMAGES_PATH.get_data_path("Fourier_collected_images_by_shape_ogyei"),
                "colour_vectors":
                    DATA_PATH.get_data_path("colour_vectors_ogyei"),
                "colour_annotated_images":
                    IMAGES_PATH.get_data_path("pill_colours_ogyei"),
                "imprint_vectors":
                    DATASET_PATH.get_data_path("imprint_vectors_ogyei"),
                "score_vectors":
                    DATASET_PATH.get_data_path("score_vectors_ogyei"),
                "concatenated_vectors":
                    DATA_PATH.get_data_path("concatenated_vectors_ogyei")
            }
        },

        # ---------------------------------------------------- N I H ---------------------------------------------------
        "cure_one_sided": {
            "customer": {
                "customer_images":
                    DATASET_PATH.get_data_path("cure_one_sided_customer_images"),
                "customer_csv":
                    DATASET_PATH.get_data_path("cure_one_sided_customer_csv"),
                "customer_xlsx":
                    DATASET_PATH.get_data_path("cure_one_sided_customer_xlsx"),
                "customer_txt":
                    DATASET_PATH.get_data_path("cure_one_sided_customer_txt"),
                "customer_mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_customer_mask_images"),
                "customer_segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_customer_segmentation_labels")
            },

            "reference": {
                "reference_images":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_images"),
                "reference_masks":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_masks"),
                "reference_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_labels"),
                "reference_csv":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_csv"),
                "reference_xlsx":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_xlsx"),
                "reference_txt":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_txt"),
                "reference_mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_mask_images"),
                "reference_segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_reference_segmentation_labels")
            },

            "train": {
                "images":
                    DATASET_PATH.get_data_path("cure_one_sided_train_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_train_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_train_segmentation_labels"),
                "aug_images":
                    DATASET_PATH.get_data_path("cure_one_sided_train_aug_images"),
                "aug_yolo_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_train_aug_yolo_labels"),
                "aug_mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_train_aug_mask_images"),
            },

            "valid": {
                "images":
                    DATASET_PATH.get_data_path("cure_one_sided_valid_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_valid_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_valid_segmentation_labels"),
                "aug_images":
                    DATASET_PATH.get_data_path("cure_one_sided_valid_aug_images"),
                "aug_yolo_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_valid_aug_yolo_labels"),
                "aug_mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_valid_aug_mask_images"),
            },

            "test": {
                "images":
                    DATASET_PATH.get_data_path("cure_one_sided_test_images"),
                "mask_images":
                    DATASET_PATH.get_data_path("cure_one_sided_test_mask_images"),
                "segmentation_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_test_segmentation_labels"),
                "aug_images":
                    DATASET_PATH.get_data_path("cure_one_sided_test_aug_images"),
                "aug_yolo_labels":
                    DATASET_PATH.get_data_path("cure_one_sided_test_aug_yolo_labels"),
            },

            "src_stream_images": {
                "reference": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_reference_texture"),
                },
                "customer": {
                    "stream_images":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_customer"),
                    "stream_images_contour":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_customer_contour"),
                    "stream_images_lbp":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_customer_lbp"),
                    "stream_images_rgb":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_customer_rgb"),
                    "stream_images_texture":
                        DATASET_PATH.get_data_path("stream_images_cure_one_sided_customer_texture"),
                }
            },

            "dst_stream_images": {
                'stream_images_anchor':
                    IMAGES_PATH.get_data_path("stream_images_cure_one_sided_anchor"),
                "stream_images_pos_neg":
                    IMAGES_PATH.get_data_path("stream_images_cure_one_sided_pos_neg"),
                'ref':
                    IMAGES_PATH.get_data_path("test_ref_cure_one_sided"),
                'query':
                    IMAGES_PATH.get_data_path("test_query_cure_one_sided")
            },

            "other": {
                'k_fold':
                    DATA_PATH.get_data_path("cure_one_sided_k_fold"),
            },

            "dynamic_margin": {
                "pill_desc_xlsx":
                    DATA_PATH.get_data_path("pill_desc_xlsx_cure_one_sided"),
                "Fourier_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_cure_one_sided"),
                "Fourier_euclidean_distance":
                    IMAGES_PATH.get_data_path("Fourier_euclidean_distance_cure_one_sided"),
                "Fourier_saved_mean_vectors":
                    DATA_PATH.get_data_path("Fourier_saved_mean_vectors_cure_one_sided"),
                "Fourier_images_by_shape":
                    IMAGES_PATH.get_data_path("Fourier_collected_images_by_shape_cure_one_sided"),
                "colour_vectors":
                    DATA_PATH.get_data_path("colour_vectors_cure_one_sided"),
                "colour_annotated_images":
                    IMAGES_PATH.get_data_path("pill_colours_cure_one_sided"),
                "imprint_vectors":
                    DATASET_PATH.get_data_path("imprint_vectors_cure_one_sided"),
                "score_vectors":
                    DATASET_PATH.get_data_path("score_vectors_cure_one_sided"),
                "concatenated_vectors":
                    DATA_PATH.get_data_path("concatenated_vectors_cure_one_sided")
            }
        },

        # ---------------------------------------------------- D T D ---------------------------------------------------
        "dtd": {
            "dataset_path": DATASET_PATH.get_data_path("dtd_images")
        }
    }

    return path_to_images[dataset_name]
