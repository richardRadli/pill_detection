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
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_contour_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_contour_ogyei_v2_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_contour_ogyei_v2_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_contour_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_contour_ogyei_v2_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei_v2_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei_v2_dmtl_textual")
                        }
                    }
                }
            },
            "cure_one_sided": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("contour_stream_cure_one_sided_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("contour_stream_cure_one_sided_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("contour_stream_ref_cure_one_sided"),
                        "query": IMAGES_PATH.get_data_path("contour_stream_query_cure_one_sided")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_contour_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_contour_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_contour_cure_one_sided_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_contour_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_contour_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_contour_cure_one_sided_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure_one_sided_dmtl_textual")
                        }
                    }
                }
            },
            "cure_two_sided": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("contour_stream_cure_two_sided_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("contour_stream_cure_two_sided_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("contour_stream_ref_cure_two_sided"),
                        "query": IMAGES_PATH.get_data_path("contour_stream_query_cure_two_sided")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_contour_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_contour_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_contour_cure_two_sided_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_contour_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_contour_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_contour_cure_two_sided_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure_two_sided_dmtl_textual")
                        }
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
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_ogyei_v2_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_ogyei_v2_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_ogyei_v2_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei_v2_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei_v2_dmtl_textual")
                        }
                    }
                }
            },
            "cure_one_sided": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("lbp_stream_cure_one_sided_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("lbp_stream_cure_one_sided_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("lbp_stream_ref_cure_one_sided"),
                        "query": IMAGES_PATH.get_data_path("lbp_stream_query_cure_one_sided")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_cure_one_sided_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_cure_one_sided_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_one_sided_dmtl_textual")
                        }
                    }
                }
            },
            "cure_two_sided": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("lbp_stream_cure_two_sided_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("lbp_stream_cure_two_sided_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("lbp_stream_ref_cure_two_sided"),
                        "query": IMAGES_PATH.get_data_path("lbp_stream_query_cure_two_sided")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_lbp_cure_two_sided_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_lbp_cure_two_sided_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_two_sided_dmtl_textual")
                        }
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
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_ogyei_v2_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_ogyei_v2_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_ogyei_v2_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei_v2_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei_v2_dmtl_textual")
                        }
                    }
                }
            },
            "cure_one_sided": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("rgb_stream_cure_one_sided_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("rgb_stream_cure_one_sided_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("rgb_stream_ref_cure_one_sided"),
                        "query": IMAGES_PATH.get_data_path("rgb_stream_query_cure_one_sided")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_cure_one_sided_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_cure_one_sided_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_one_sided_dmtl_textual")
                        }
                    }
                }
            },
            "cure_two_sided": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("rgb_stream_cure_two_sided_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("rgb_stream_cure_two_sided_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("rgb_stream_ref_cure_two_sided"),
                        "query": IMAGES_PATH.get_data_path("rgb_stream_query_cure_two_sided")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_rgb_cure_two_sided_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_rgb_cure_two_sided_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_two_sided_dmtl_textual")
                        }
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
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_texture_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_texture_ogyei_v2_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_texture_ogyei_v2_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_texture_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_texture_ogyei_v2_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei_v2_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei_v2_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei_v2_dmtl_textual")
                        }
                    }
                }
            },
            "cure_one_sided": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("texture_stream_cure_one_sided_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("texture_stream_cure_one_sided_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("texture_stream_ref_cure_one_sided"),
                        "query": IMAGES_PATH.get_data_path("texture_stream_query_cure_one_sided")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_texture_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_texture_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_texture_cure_one_sided_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_texture_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_texture_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_texture_cure_one_sided_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_one_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_one_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_one_sided_dmtl_textual")
                        }
                    }
                }
            },
            "cure_two_sided": {
                "EfficientNetV2": {
                    "train": {
                        "anchor": IMAGES_PATH.get_data_path("texture_stream_cure_two_sided_anchor"),
                        "pos_neg": IMAGES_PATH.get_data_path("texture_stream_cure_two_sided_pos_neg")
                    },
                    "test": {
                        "ref": IMAGES_PATH.get_data_path("texture_stream_ref_cure_two_sided"),
                        "query": IMAGES_PATH.get_data_path("texture_stream_query_cure_two_sided")
                    },
                    "model_weights_dir": {
                        "hmtl": DATA_PATH.get_data_path("weights_efficient_net_v2_texture_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_texture_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("weights_efficient_net_v2_texture_cure_two_sided_dmtl_textual")
                        }
                    },
                    "logs_dir": {
                        "hmtl": DATA_PATH.get_data_path("logs_efficient_net_v2_texture_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_texture_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("logs_efficient_net_v2_texture_cure_two_sided_dmtl_textual")
                        }
                    },
                    "hardest_samples": {
                        "hmtl": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_two_sided_hmtl"),
                        "dmtl": {
                            "visual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_two_sided_dmtl_visual"),
                            "textual":
                                DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_two_sided_dmtl_textual")
                        }
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
                    "dmtl": {
                        "visual": DATA_PATH.get_data_path("predictions_efficient_net_v2_ogyei_v2_dmtl_visual"),
                        "textual": DATA_PATH.get_data_path("predictions_efficient_net_v2_ogyei_v2_dmtl_textual")
                    }
                },
                'plotting_folder': {
                    "hmtl": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_ogyei_v2_hmtl"),
                    "dmtl": {
                        "visual": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_ogyei_v2_dmtl_visual"),
                        "textual": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_ogyei_v2_dmtl_textual")
                    }
                },
                'ref_vectors_folder': {
                    "hmtl": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_ogyei_v2_hmtl"),
                    "dmtl": {
                        "visual": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_ogyei_v2_dmtl_visual"),
                        "textual": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_ogyei_v2_dmtl_textual")
                    }
                },
                'hard_sample': {
                    "hmtl": {
                        "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei_v2_hmtl"),
                        "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei_v2_hmtl"),
                        "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei_v2_hmtl"),
                        "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei_v2_hmtl")
                    },
                    "dmtl": {
                        "visual": {
                            "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_ogyei_v2_dmtl_visual"),
                            "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei_v2_dmtl_visual"),
                            "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei_v2_dmtl_visual"),
                            "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei_v2_dmtl_visual")
                        },
                        "textual": {
                            "Contour": DATA_PATH.get_data_path(
                                "hardest_samples_efficient_net_v2_contour_ogyei_v2_dmtl_textual"),
                            "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_ogyei_v2_dmtl_textual"),
                            "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_ogyei_v2_dmtl_textual"),
                            "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_ogyei_v2_dmtl_textual")
                        }
                    }
                }
            }
        },
        'cure_one_sided': {
            'EfficientNetV2': {
                'prediction_folder': {
                    "hmtl": DATA_PATH.get_data_path("predictions_efficient_net_v2_cure_one_sided_hmtl"),
                    "dmtl": {
                        "visual": DATA_PATH.get_data_path("predictions_efficient_net_v2_cure_one_sided_dmtl_visual"),
                        "textual": DATA_PATH.get_data_path("predictions_efficient_net_v2_cure_one_sided_dmtl_textual")
                    }
                },
                'plotting_folder': {
                    "hmtl": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_cure_one_sided_hmtl"),
                    "dmtl": {
                        "visual": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_cure_one_sided_dmtl_visual"),
                        "textual": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_cure_one_sided_dmtl_textual")
                    }
                },
                'ref_vectors_folder': {
                    "hmtl": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_cure_one_sided_hmtl"),
                    "dmtl": {
                        "visual": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_cure_one_sided_dmtl_visual"),
                        "textual": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_cure_one_sided_dmtl_textual")
                    }
                },
                'hard_sample': {
                    "hmtl": {
                        "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure_one_sided_hmtl"),
                        "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_one_sided_hmtl"),
                        "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_one_sided_hmtl"),
                        "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_one_sided_hmtl")
                    },
                    "dmtl": {
                        "visual": {
                            "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure_one_sided_dmtl_visual"),
                            "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_one_sided_dmtl_visual"),
                            "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_one_sided_dmtl_visual"),
                            "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_one_sided_dmtl_visual")
                        },
                        "textual": {
                            "Contour": DATA_PATH.get_data_path(
                                "hardest_samples_efficient_net_v2_contour_cure_one_sided_dmtl_textual"),
                            "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_one_sided_dmtl_textual"),
                            "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_one_sided_dmtl_textual"),
                            "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_one_sided_dmtl_textual")
                        }
                    }
                }
            }
        },
        'cure_two_sided': {
            'EfficientNetV2': {
                'prediction_folder': {
                    "hmtl": DATA_PATH.get_data_path("predictions_efficient_net_v2_cure_two_sided_hmtl"),
                    "dmtl": {
                        "visual": DATA_PATH.get_data_path("predictions_efficient_net_v2_cure_two_sided_dmtl_visual"),
                        "textual": DATA_PATH.get_data_path("predictions_efficient_net_v2_cure_two_sided_dmtl_textual")
                    }
                },
                'plotting_folder': {
                    "hmtl": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_cure_two_sided_hmtl"),
                    "dmtl": {
                        "visual": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_cure_two_sided_dmtl_visual"),
                        "textual": IMAGES_PATH.get_data_path("plotting_efficient_net_v2_cure_two_sided_dmtl_textual")
                    }
                },
                'ref_vectors_folder': {
                    "hmtl": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_cure_two_sided_hmtl"),
                    "dmtl": {
                        "visual": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_cure_two_sided_dmtl_visual"),
                        "textual": DATA_PATH.get_data_path("reference_vectors_efficient_net_v2_cure_two_sided_dmtl_textual")
                    }
                },
                'hard_sample': {
                    "hmtl": {
                        "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure_two_sided_hmtl"),
                        "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_two_sided_hmtl"),
                        "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_two_sided_hmtl"),
                        "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_two_sided_hmtl")
                    },
                    "dmtl": {
                        "visual": {
                            "Contour": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_contour_cure_two_sided_dmtl_visual"),
                            "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_two_sided_dmtl_visual"),
                            "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_two_sided_dmtl_visual"),
                            "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_two_sided_dmtl_visual")
                        },
                        "textual": {
                            "Contour": DATA_PATH.get_data_path(
                                "hardest_samples_efficient_net_v2_contour_cure_two_sided_dmtl_textual"),
                            "LBP": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_lbp_cure_two_sided_dmtl_textual"),
                            "RGB": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_rgb_cure_two_sided_dmtl_textual"),
                            "Texture": DATA_PATH.get_data_path("hardest_samples_efficient_net_v2_texture_cure_two_sided_dmtl_textual")
                        }
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
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "logs_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "logs_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_textual"
                            )
                    }
                },
                'weights_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "weights_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_textual"
                            )
                    }
                },
                'prediction_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "predictions_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_textual"
                            )
                    }
                },
                'plotting_folder': {
                    "hmtl":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "plotting_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_textual"
                            )
                    }
                },
                'ref_vectors_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "ref_vec_fusion_network_efficient_net_v2_multihead_attention_ogyei_v2_dmtl_textual"
                            )
                    }
                }
            }
        },
        'cure_one_sided': {
            'EfficientNetV2MultiHeadAttention': {
                'logs_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "logs_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "logs_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "logs_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_textual"
                            )
                    }
                },
                'weights_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "weights_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "weights_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "weights_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_textual"
                            )
                    }
                },
                'prediction_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_textual"
                            )
                    }
                },
                'plotting_folder': {
                    "hmtl":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_textual"
                            )
                    }
                },
                'ref_vectors_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_one_sided_dmtl_textual"
                            )
                    }
                }
            }
        },
        'cure_two_sided': {
            'EfficientNetV2MultiHeadAttention': {
                'logs_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "logs_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "logs_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "logs_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_textual"
                            )
                    }
                },
                'weights_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "weights_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "weights_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "weights_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_textual"
                            )
                    }
                },
                'prediction_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "predictions_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_textual"
                            )
                    }
                },
                'plotting_folder': {
                    "hmtl":
                        IMAGES_PATH.get_data_path(
                            "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "plotting_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_textual"
                            )
                    }
                },
                'ref_vectors_folder': {
                    "hmtl":
                        DATA_PATH.get_data_path(
                            "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_hmtl"
                        ),
                    "dmtl": {
                        "visual":
                            DATA_PATH.get_data_path(
                                "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_visual"
                            ),
                        "textual":
                            DATA_PATH.get_data_path(
                                "ref_vec_fusion_network_efficient_net_v2_multihead_attention_cure_two_sided_dmtl_textual"
                            )
                    }
                }
            }
        }
    }

    return network_configs[dataset_type][network_type]


def word_embedding_paths(dataset_type):
    word_embedding_config = {
        "cure_one_sided": {
            "emb_euc_mtx":
                IMAGES_PATH.get_data_path("emb_euc_mtx_word_embedded_network_cure_one_sided"),
            "emb_tsne":
                IMAGES_PATH.get_data_path("emb_tsne_word_embedded_network_cure_one_sided"),
            "weights":
                DATA_PATH.get_data_path("weights_word_embedded_network_cure_one_sided"),
            "logs":
                DATA_PATH.get_data_path("logs_word_embedded_network_cure_one_sided"),
            "predictions":
                DATA_PATH.get_data_path("predictions_word_embedded_network_cure_one_sided")
        },
        "cure_two_sided": {
            "emb_euc_mtx":
                IMAGES_PATH.get_data_path("emb_euc_mtx_word_embedded_network_cure_two_sided"),
            "emb_tsne":
                IMAGES_PATH.get_data_path("emb_tsne_word_embedded_network_cure_two_sided"),
            "weights":
                DATA_PATH.get_data_path("weights_word_embedded_network_cure_two_sided"),
            "logs":
                DATA_PATH.get_data_path("logs_word_embedded_network_cure_two_sided"),
            "predictions":
                DATA_PATH.get_data_path("predictions_word_embedded_network_cure_two_sided")
        },
        "ogyeiv2": {
            "emb_euc_mtx":
                IMAGES_PATH.get_data_path("emb_euc_mtx_word_embedded_network_ogyei_v2"),
            "emb_tsne":
                IMAGES_PATH.get_data_path("emb_tsne_word_embedded_network_ogyei_v2"),
            "weights":
                DATA_PATH.get_data_path("weights_word_embedded_network_ogyei_v2"),
            "logs":
                DATA_PATH.get_data_path("logs_word_embedded_network_ogyei_v2"),
            "predictions":
                DATA_PATH.get_data_path("predictions_word_embedded_network_ogyei_v2")
        }
    }

    return word_embedding_config[dataset_type]
