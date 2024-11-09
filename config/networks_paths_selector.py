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
                "CNN": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("train_contour_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("contour_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("contour_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("contour_stream_query_cure")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_cnn_contour_cure"),
                    "logs_dir": DATA_PATH.get_data_path("logs_cnn_contour_cure"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_cnn_contour_cure")
                },
                "EfficientNet": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("train_contour_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("contour_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("contour_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("contour_stream_query_cure")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_efficient_net_contour_cure"),
                    "logs_dir": DATA_PATH.get_data_path("logs_efficient_net_contour_cure"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_cure")
                }
            },
            "ogyei": {
                "CNN": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("contour_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("contour_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("contour_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("contour_stream_query_ogyei")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_cnn_contour_ogyei"),
                    "logs_dir": DATA_PATH.get_data_path("logs_cnn_contour_ogyei"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_cnn_contour_ogyei")
                },
                "EfficientNet": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("contour_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("contour_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("contour_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("contour_stream_query_ogyei")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_efficient_net_contour_ogyei"),
                    "logs_dir": DATA_PATH.get_data_path("logs_efficient_net_contour_ogyei"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_ogyei")
                }
            }
        },
        # ----------------------------------------------- L B P ------------------------------------------------
        "LBP": {
            "cure": {
                "CNN": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("lbp_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("lbp_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("lbp_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("lbp_stream_query_cure")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_cnn_lbp_cure"),
                    "logs_dir": DATA_PATH.get_data_path("logs_cnn_lbp_cure"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_cnn_lbp_cure")
                },
                "EfficientNet": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("lbp_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("lbp_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("lbp_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("lbp_stream_query_cure")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_efficient_net_lbp_cure"),
                    "logs_dir": DATA_PATH.get_data_path("logs_efficient_net_lbp_cure"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_cure")
                }
            },
            "ogyei": {
                "CNN": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("lbp_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("lbp_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("lbp_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("lbp_stream_query_ogyei")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_cnn_lbp_ogyei"),
                    "logs_dir": DATA_PATH.get_data_path("logs_cnn_lbp_ogyei"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_cnn_lbp_ogyei")
                },
                "EfficientNet": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("lbp_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("lbp_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("lbp_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("lbp_stream_query_ogyei")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_efficient_net_lbp_ogyei"),
                    "logs_dir": DATA_PATH.get_data_path("logs_efficient_net_lbp_ogyei"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_ogyei")
                }
            }
        },
        # ----------------------------------------------- R G B ------------------------------------------------
        "RGB": {
            "cure": {
                "CNN": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("rgb_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("rgb_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("rgb_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("rgb_stream_query_cure")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_cnn_rgb_cure"),
                    "logs_dir": DATA_PATH.get_data_path("logs_cnn_rgb_cure"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_cnn_rgb_cure")
                },
                "EfficientNet": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("rgb_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("rgb_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("rgb_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("rgb_stream_query_cure")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_efficient_net_rgb_cure"),
                    "logs_dir": DATA_PATH.get_data_path("logs_efficient_net_rgb_cure"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_cure")
                }
            },
            "ogyei": {
                "CNN": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("rgb_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("rgb_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("rgb_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("rgb_stream_query_ogyei")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_cnn_rgb_ogyei"),
                    "logs_dir": DATA_PATH.get_data_path("logs_cnn_rgb_ogyei"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_cnn_rgb_ogyei")
                },
                "EfficientNet": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("rgb_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("rgb_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("rgb_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("rgb_stream_query_ogyei")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_efficient_net_rgb_ogyei"),
                    "logs_dir": DATA_PATH.get_data_path("logs_efficient_net_rgb_ogyei"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_ogyei")
                }
            }
        },
        # -------------------------------------------- T E X T U R E ------------------------------------------------
        "Texture": {
            "cure": {
                "CNN": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("texture_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("texture_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("texture_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("texture_stream_query_cure")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_cnn_texture_cure"),
                    "logs_dir": DATA_PATH.get_data_path("logs_cnn_texture_cure"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_cnn_texture_cure")
                },
                "EfficientNet": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("texture_stream_cure_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("texture_stream_cure_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("texture_stream_ref_cure"),
                        "query": IMAGES_PATH.get_data_path("texture_stream_query_cure")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_efficient_net_texture_cure"),
                    "logs_dir": DATA_PATH.get_data_path("logs_efficient_net_texture_cure"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_cure")
                }
            },
            "ogyei": {
                "CNN": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("texture_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("texture_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("texture_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("texture_stream_query_ogyei")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_cnn_texture_ogyei"),
                    "logs_dir": DATA_PATH.get_data_path("logs_cnn_texture_ogyei"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_cnn_texture_ogyei")
                },
                "EfficientNet": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("texture_stream_ogyei_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("texture_stream_ogyei_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("texture_stream_ref_ogyei"),
                        "query": IMAGES_PATH.get_data_path("texture_stream_query_ogyei")
                    },
                    "model_weights_dir": DATA_PATH.get_data_path("weights_efficient_net_texture_ogyei"),
                    "logs_dir": DATA_PATH.get_data_path("logs_efficient_net_texture_ogyei"),
                    "hardest_samples": DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_ogyei")
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
        network_type (str): The type of network, e.g., 'CNN' or 'EfficientNet'.

    Returns:
         dict: Dictionary containing the prediction, plotting, and reference vectors folder paths.
    """

    network_configs = {
        'cure': {
            'CNN': {
                'prediction_folder': DATA_PATH.get_data_path("predictions_cnn_cure"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_cnn_cure"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_cnn_cure"),
                'hard_sample': {
                    "Contour": DATA_PATH.get_data_path("hardest_samples_cnn_contour_cure"),
                    "LBP": DATA_PATH.get_data_path("hardest_samples_cnn_lbp_cure"),
                    "RGB": DATA_PATH.get_data_path("hardest_samples_cnn_rgb_cure"),
                    "Texture": DATA_PATH.get_data_path("hardest_samples_cnn_texture_cure")
                }
            },
            'EfficientNet': {
                'prediction_folder': DATA_PATH.get_data_path("predictions_efficient_net_cure"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_efficient_net_cure"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_efficient_net_cure"),
                'hard_sample': {
                    "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_cure"),
                    "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_cure"),
                    "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_cure"),
                    "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_cure")
                }
            }
        },
        'ogyei': {
            'CNN': {
                'prediction_folder': DATA_PATH.get_data_path("predictions_cnn_ogyei"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_cnn_ogyei"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_cnn_ogyei"),
                'hard_sample': {
                    "Contour": DATA_PATH.get_data_path("hardest_samples_cnn_contour_ogyei"),
                    "LBP": DATA_PATH.get_data_path("hardest_samples_cnn_lbp_ogyei"),
                    "RGB": DATA_PATH.get_data_path("hardest_samples_cnn_rgb_ogyei"),
                    "Texture": DATA_PATH.get_data_path("hardest_samples_cnn_texture_ogyei")
                }
            },
            'EfficientNet': {
                'prediction_folder': DATA_PATH.get_data_path("predictions_efficient_net_ogyei"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_efficient_net_ogyei"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_efficient_net_ogyei"),
                'hard_sample': {
                    "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_contour_ogyei"),
                    "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_lbp_ogyei"),
                    "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_rgb_ogyei"),
                    "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_texture_ogyei")
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
            'CNNFusionNet': {
                'logs_folder': DATA_PATH.get_data_path("logs_fusion_network_cnn_cure"),
                'weights_folder': DATA_PATH.get_data_path("weights_fusion_network_cnn_cure"),
                'prediction_folder': DATA_PATH.get_data_path("predictions_fusion_network_cnn_cure"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_fusion_network_cnn_cure"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_fusion_network_cnn_cure")
            },
            'EfficientNetSelfAttention': {
                'logs_folder': DATA_PATH.get_data_path("logs_fusion_network_efficient_net_self_attention_cure"),
                'weights_folder': DATA_PATH.get_data_path("weights_fusion_network_efficient_net_self_attention_cure"),
                'prediction_folder':
                    DATA_PATH.get_data_path("predictions_fusion_network_efficient_net_self_attention_cure"),
                'plotting_folder':
                    IMAGES_PATH.get_data_path("plotting_fusion_network_efficient_net_self_attention_cure"),
                'ref_vectors_folder':
                    DATA_PATH.get_data_path("reference_vectors_fusion_network_efficient_net_self_attention_cure")
            }
        },
        'ogyei': {
            'CNNFusionNet': {
                'logs_folder': DATA_PATH.get_data_path("logs_fusion_network_cnn_ogyei"),
                'weights_folder': DATA_PATH.get_data_path("weights_fusion_network_cnn_ogyei"),
                'prediction_folder': DATA_PATH.get_data_path("predictions_fusion_network_cnn_ogyei"),
                'plotting_folder': IMAGES_PATH.get_data_path("plotting_fusion_network_cnn_ogyei"),
                'ref_vectors_folder': DATA_PATH.get_data_path("reference_vectors_fusion_network_cnn_ogyei")
            },
            'EfficientNetSelfAttention': {
                'logs_folder': DATA_PATH.get_data_path("logs_fusion_network_efficient_net_self_attention_ogyei"),
                'weights_folder': DATA_PATH.get_data_path("weights_fusion_network_efficient_net_self_attention_ogyei"),
                'prediction_folder':
                    DATA_PATH.get_data_path("predictions_fusion_network_efficient_net_self_attention_ogyei"),
                'plotting_folder':
                    IMAGES_PATH.get_data_path("plotting_fusion_network_efficient_net_self_attention_ogyei"),
                'ref_vectors_folder':
                    DATA_PATH.get_data_path("reference_vectors_fusion_network_efficient_net_self_attention_ogyei")
            }
        }
    }

    return network_configs[dataset_type][network_type]


def segmentation_paths(network_type, dataset_name):
    network_paths = {
        "UNet": {
            "cure": {
                'logs_folder':
                    DATA_PATH.get_data_path("logs_unet_cure"),
                'weights_folder':
                    DATA_PATH.get_data_path("weights_unet_cure"),
                'prediction_folder':
                    {
                        "out":
                            IMAGES_PATH.get_data_path("unet_out_cure"),
                        "compare":
                            IMAGES_PATH.get_data_path("unet_compare_cure"),
                    }
            },
            "ogyei": {
                'logs_folder':
                    DATA_PATH.get_data_path("logs_unet_ogyei"),
                'weights_folder':
                    DATA_PATH.get_data_path("weights_unet_ogyei"),
                'prediction_folder':
                {
                    "out":
                        IMAGES_PATH.get_data_path("unet_out_ogyei"),
                    "compare":
                        IMAGES_PATH.get_data_path("unet_compare_ogyei"),
                }
            }
        },
        "W2Net": {
            "cure": {
                'logs_folder':
                    DATA_PATH.get_data_path("logs_w2net_cure"),
                'weights_folder':
                    DATA_PATH.get_data_path("weights_w2net_cure"),
                'prediction_folder':
                {
                    "out":
                        IMAGES_PATH.get_data_path("w2net_out_cure"),
                    "compare":
                        IMAGES_PATH.get_data_path("w2net_compare_cure"),
                }
            },
            "ogyei": {
                'logs_folder':
                    DATA_PATH.get_data_path("logs_w2net_ogyei"),
                'weights_folder':
                    DATA_PATH.get_data_path("weights_w2net_ogyei"),
                'prediction_folder':
                {
                    "out":
                        IMAGES_PATH.get_data_path("w2net_out_ogyei"),
                    "compare":
                        IMAGES_PATH.get_data_path("w2net_compare_ogyei"),
                }
            }
        }
    }
    return network_paths[network_type][dataset_name]
