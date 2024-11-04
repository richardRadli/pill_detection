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
            "cure": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("contour_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("contour_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("contour_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("contour_stream_query_cure")
                    },
                    "model_weights_dir":
                        DATA_PATH.get_data_path("weights_efficient_net_v2_contour_cure"),
                    "logs_dir":
                        DATA_PATH.get_data_path("logs_efficient_net_v2_contour_cure"),
                    "hardest_samples":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure")
                }
            },
            "ogyei": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("contour_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("contour_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("contour_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("contour_stream_query_ogyei")
                    },
                    "model_weights_dir":
                        DATA_PATH.get_data_path("weights_efficient_net_v2_contour_ogyei"),
                    "logs_dir":
                        DATA_PATH.get_data_path("logs_efficient_net_v2_contour_ogyei"),
                    "hardest_samples":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei")
                }
            }
        },
        # ----------------------------------------------- L B P ------------------------------------------------
        "LBP": {
            "cure": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("lbp_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("lbp_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("lbp_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("lbp_stream_query_cure")
                    },
                    "model_weights_dir":
                        DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_cure"),
                    "logs_dir":
                        DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_cure"),
                    "hardest_samples":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure")
                }
            },
            "ogyei": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("lbp_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("lbp_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("lbp_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("lbp_stream_query_ogyei")
                    },
                    "model_weights_dir":
                        DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_ogyei"),
                    "logs_dir":
                        DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_ogyei"),
                    "hardest_samples":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei")
                }
            }
        },
        # ----------------------------------------------- R G B ------------------------------------------------
        "RGB": {
            "cure": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("rgb_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("rgb_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("rgb_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("rgb_stream_query_cure")
                    },
                    "model_weights_dir":
                        DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_cure"),
                    "logs_dir":
                        DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_cure"),
                    "hardest_samples":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure")
                }
            },
            "ogyei": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("rgb_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("rgb_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("rgb_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("rgb_stream_query_ogyei")
                    },
                    "model_weights_dir":
                        DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_ogyei"),
                    "logs_dir":
                        DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_ogyei"),
                    "hardest_samples":
                        DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei")
                }
            }
        },
        # -------------------------------------------- T E X T U R E ------------------------------------------------
        "Texture": {
            "cure": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("texture_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("texture_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("texture_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("texture_stream_query_cure")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_efficient_net_v2_texture_cure"),
                    "logs_dir": DATA_PATH.get_data_path("logs_efficient_net_v2_texture_cure"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure")
                }
            },
            "ogyei": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("texture_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("texture_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("texture_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("texture_stream_query_ogyei")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_efficient_net_v2_texture_ogyei"),
                    "logs_dir": DATA_PATH.get_data_path("logs_efficient_net_v2_texture_ogyei"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei")
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
        'cure': {
            'EfficientNetV2': {
                'prediction_folder': DATA_PATH.get_data_path("predictions_efficient_net_v2_cure"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_efficient_net_v2_cure"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_cure"),
                'hard_sample': {
                    "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure"),
                    "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure"),
                    "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure"),
                    "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure")
                }
            }
        },
        'ogyei': {
            'EfficientNetV2': {
                'prediction_folder': DATA_PATH.get_data_path("predictions_efficient_net_v2_ogyei"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_efficient_net_v2_ogyei"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_ogyei"),
                'hard_sample': {
                    "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei"),
                    "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei"),
                    "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei"),
                    "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei")
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
        'cure': {
            'EfficientNetV2SelfAttention': {
                'logs_folder':
                    DATA_PATH.get_data_path("logs_fusion_network_efficient_net_v2_self_attention_cure"),
                'weights_folder':
                    DATA_PATH.get_data_path("weights_fusion_network_efficient_net_v2_self_attention_cure"),
                'prediction_folder':
                    DATA_PATH.get_data_path("predictions_fusion_network_efficient_net_v2_self_attention_cure"),
                'plotting_folder':
                    IMAGES_PATH.get_data_path("plotting_fusion_network_efficient_net_v2_self_attention_cure"),
                'ref_vectors_folder':
                    DATA_PATH.get_data_path("ref_vec_fusion_network_efficient_net_v2_self_attention_cure")
            },
            'EfficientNetV2MultiHeadAttention': {
                'logs_folder':
                    DATA_PATH.get_data_path("logs_fusion_network_efficient_net_v2_multihead_attention_cure"),
                'weights_folder':
                    DATA_PATH.get_data_path("weights_fusion_network_efficient_net_v2_multihead_attention_cure"),
                'prediction_folder':
                    DATA_PATH.get_data_path("predictions_fusion_network_efficient_net_v2_multihead_attention_cure"),
                'plotting_folder':
                    IMAGES_PATH.get_data_path("plotting_fusion_network_efficient_net_v2_multihead_attention_cure"),
                'ref_vectors_folder':
                    DATA_PATH.get_data_path("ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure")
            },
            'EfficientNetV2MHAFMHA': {
                'logs_folder':
                    DATA_PATH.get_data_path("logs_fusion_network_efficient_net_v2_MHAFMHA_cure"),
                'weights_folder':
                    DATA_PATH.get_data_path("weights_fusion_network_efficient_net_v2_MHAFMHA_cure"),
                'prediction_folder':
                    DATA_PATH.get_data_path("predictions_fusion_network_efficient_net_v2_MHAFMHA_cure"),
                'plotting_folder':
                    IMAGES_PATH.get_data_path("plotting_fusion_network_efficient_net_v2_MHAFMHA_cure"),
                'ref_vectors_folder':
                    DATA_PATH.get_data_path("ref_vec_fusion_network_efficient_net_v2_MHAFMHA_cure")
            }
        },
        'ogyei': {
            'EfficientNetV2SelfAttention': {
                'logs_folder':
                    DATA_PATH.get_data_path("logs_fusion_network_efficient_net_v2_self_attention_ogyei"),
                'weights_folder':
                    DATA_PATH.get_data_path("weights_fusion_network_efficient_net_v2_self_attention_ogyei"),
                'prediction_folder':
                    DATA_PATH.get_data_path("predictions_fusion_network_efficient_net_v2_self_attention_ogyei"),
                'plotting_folder':
                    IMAGES_PATH.get_data_path("plotting_fusion_network_efficient_net_v2_self_attention_ogyei"),
                'ref_vectors_folder':
                    DATA_PATH.get_data_path("ref_vec_fusion_network_efficient_net_v2_self_attention_ogyei")
            },
            'EfficientNetV2MultiHeadAttention': {
                'logs_folder':
                    DATA_PATH.get_data_path("logs_fusion_network_efficient_net_v2_multihead_attention_ogyei"),
                'weights_folder':
                    DATA_PATH.get_data_path("weights_fusion_network_efficient_net_v2_multihead_attention_ogyei"),
                'prediction_folder':
                    DATA_PATH.get_data_path("predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei"),
                'plotting_folder':
                    IMAGES_PATH.get_data_path("plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei"),
                'ref_vectors_folder':
                    DATA_PATH.get_data_path("ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei")
            },
            'EfficientNetV2MHAFMHA': {
                'logs_folder':
                    DATA_PATH.get_data_path("logs_fusion_network_efficient_net_v2_MHAFMHA_ogyei"),
                'weights_folder':
                    DATA_PATH.get_data_path("weights_fusion_network_efficient_net_v2_MHAFMHA_ogyei"),
                'prediction_folder':
                    DATA_PATH.get_data_path("predictions_fusion_network_efficient_net_v2_MHAFMHA_ogyei"),
                'plotting_folder':
                    IMAGES_PATH.get_data_path("plotting_fusion_network_efficient_net_v2_MHAFMHA_ogyei"),
                'ref_vectors_folder':
                    DATA_PATH.get_data_path("ref_vec_fusion_network_efficient_net_v2_MHAFMHA_ogyei")
            }
        }
    }

    return network_configs[dataset_type][network_type]
