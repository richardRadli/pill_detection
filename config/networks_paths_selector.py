from typing import Dict

from config.data_paths import DATA_PATH, IMAGES_PATH


def substream_paths() -> Dict:
    """
    Returns the dictionary containing the configuration details for the four different subnetworks
    (Contour, LBP, RGB, Texture) used in the StreamNetwork phase.

    Returns:
         A dictionary containing the configuration details for the four subnetworks.
    """

    network_config = {
        # ----------------------------------------------- C O N T O U R ------------------------------------------------
        "Contour": {
            "ogyeiv2": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("contour_stream_ogyei_v2_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("contour_stream_ogyei_v2_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("contour_stream_ref_ogyei_v2"),
                        "query": IMAGES_PATH.get_data_path("contour_stream_query_ogyei_v2")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_contour_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_contour_ogyei_v2_dmtl")
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_contour_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_contour_ogyei_v2_dmtl"),
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei_v2_dmtl")
                    }
                }
            }
        },
        # ----------------------------------------------- L B P ------------------------------------------------
        "LBP": {
            "ogyeiv2": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("lbp_stream_ogyei_v2_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("lbp_stream_ogyei_v2_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("lbp_stream_ref_ogyei_v2"),
                        "query": IMAGES_PATH.get_data_path("lbp_stream_query_ogyei_v2")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_ogyei_v2_dmtl")
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_ogyei_v2_dmtl"),
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei_v2_dmtl")
                    }
                }
            }
        },
        # ----------------------------------------------- R G B ------------------------------------------------
        "RGB": {
            "ogyeiv2": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("rgb_stream_ogyei_v2_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("rgb_stream_ogyei_v2_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("rgb_stream_ref_ogyei_v2"),
                        "query": IMAGES_PATH.get_data_path("rgb_stream_query_ogyei_v2")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_ogyei_v2_dmtl")
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_ogyei_v2_dmtl"),
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei_v2_dmtl")
                    }
                }
            }
        },
        # -------------------------------------------- T E X T U R E ------------------------------------------------
        "Texture": {
            "ogyeiv2": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("texture_stream_ogyei_v2_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("texture_stream_ogyei_v2_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("texture_stream_ref_ogyei_v2"),
                        "query": IMAGES_PATH.get_data_path("texture_stream_query_ogyei_v2")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_texture_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_texture_ogyei_v2_dmtl")
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_texture_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_texture_ogyei_v2_dmtl"),
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei_v2_hmtl"),
                        "dmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei_v2_dmtl")
                    }
                }
            }
        }
    }

    return network_config


# ------------------------------------------------------------------------------------------------------------------
# ---------------------------------- G E T   M A I N   N E T W O R K   C O N F I G ---------------------------------
# ------------------------------------------------------------------------------------------------------------------
def stream_network_backbone_paths(dataset_type, network_type) -> Dict:
    """
    Returns a dictionary containing the prediction, plotting, and reference vectors folder paths for different types of
    networks, organized first by dataset and then by network type.

    Args:
        dataset_type (str): The type of dataset to use.
        network_type (str): The type of network, e.g., 'CNN' or 'EfficientNetV2'.

    Returns:
         dict: Dictionary containing the prediction, plotting, and reference vectors folder paths.
    """

    network_configs = {
        'ogyeiv2': {
            'EfficientNetV2': {
                'prediction_folder': {
                    "hmtl": DATA_PATH.get_data_path("predictions_efficient_net_v2_ogyei_v2_hmtl"),
                    "dmtl": DATA_PATH.get_data_path("predictions_efficient_net_v2_ogyei_v2_dmtl")
                },
                'plotting_folder': {
                    "hmtl": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_ogyei_v2_hmtl"),
                    "dmtl": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_ogyei_v2_dmtl")
                },
                'ref_vectors_folder': {
                    "hmtl": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_ogyei_v2_hmtl"),
                    "dmtl": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_ogyei_v2_dmtl"),
                },
                'hard_sample': {
                    "hmtl": {
                        "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei_v2_hmtl"),
                        "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei_v2_hmtl"),
                        "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei_v2_hmtl"),
                        "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei_v2_hmtl")
                    },
                    "dmtl": {
                        "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei_v2_dmtl"),
                        "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei_v2_dmtl"),
                        "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei_v2_dmtl"),
                        "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei_v2_dmtl")
                    }
                }
            }
        }
    }

    return network_configs[dataset_type][network_type]


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------ M A I N   N E T W O R K   C O N F I G   F U S I O N   T R A I N I N G -----------------------
# ----------------------------------------------------------------------------------------------------------------------
def fusion_network_paths(dataset_type: str, network_type: str) -> Dict:
    """
    Returns a dictionary containing the logs, weights, predictions, plotting, and reference vectors folder paths for
    different fusion networks based on the network_type parameter.

    Args:
        dataset_type (str): The type of dataset.
        network_type (str): The type of fusion network, e.g., 'CNNFusionNet' or 'EfficientNetSelfAttention'.

    Returns:
        dict: Dictionary containing the folder paths for logs, weights, predictions, plotting, and reference vectors.
    """

    network_configs = {
        'ogyeiv2': {
            'EfficientNetV2MultiHeadAttention': {
                'logs_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "logs_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl"
                        ),
                    "dmtl":
                        DATA_PATH.get_data_path(
                            "logs_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl"
                        ),
                },
                'weights_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl"
                        ),
                    "dmtl":
                        DATA_PATH.get_data_path(
                            "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl"
                        )
                },
                'prediction_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl"
                        ),
                    "dmtl":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl"
                        ),
                },
                'plotting_folder': {
                    "hmtl":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl"
                        ),
                    "dmtl":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl"
                        ),
                },
                'ref_vectors_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl"
                        ),
                    "dmtl":
                        DATA_PATH.get_data_path(
                            "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl"
                        )
                }
            }
        }
    }

    return network_configs[dataset_type][network_type]
