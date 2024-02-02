"""
File: fusion_network_selector.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 26, 2023

Description:
This program defines a set of wrapper classes and a factory class for creating different fusion network
models. The BaseNetwork class is an abstract base class that defines the interface for a network model.
"""

import torch

from abc import ABC, abstractmethod
from typing import Optional

from fusion_network_models.efficient_net_multihead_attention import EfficientNetMultiHeadAttention


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++ B A S E   N E T W O R K ++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class BaseNetwork(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def forward(self, x):
        pass


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++ E F F I C I E N T N E T   S E L F   A T T E N T I O N   W R A P P E R ++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EfficientNetMultiHeadAttentionWrapper(BaseNetwork):
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture):
        self.model = (
            EfficientNetMultiHeadAttention(type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb,
                                           network_cfg_texture))

    def forward(self, x):
        return self.model(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++ N E T   F A C T O R Y +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NetworkFactory:
    @staticmethod
    def create_network(fusion_network_type: str, type_of_net: str, network_cfg_contour: dict, network_cfg_lbp: dict,
                       network_cfg_rgb: dict, network_cfg_texture: dict, device: Optional[torch.device] = None) \
            -> torch.nn.Module:
        """
        Create a fusion network model based on the given fusion network type.

        :param fusion_network_type: The type of fusion network to create.
        :param type_of_net: The type of network to use within the fusion network.
        :param network_cfg_contour: Configuration for the contour network.
        :param network_cfg_lbp: Configuration for the LBP network.
        :param network_cfg_rgb: Configuration for the RGB network.
        :param network_cfg_texture: Configuration for the texture network.
        :param device: The device to use for the model (default: GPU if available, otherwise CPU).
        :return: The created fusion network model.
        """

        if fusion_network_type == "EfficientNetMultiHeadAttention":
            model = EfficientNetMultiHeadAttentionWrapper(type_of_net, network_cfg_contour, network_cfg_lbp,
                                                          network_cfg_rgb, network_cfg_texture).model
        else:
            raise ValueError("Wrong type was given!")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        return model
