"""
File: fusion_network_selector.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 26, 2023

Description:
This program defines a set of wrapper classes and a factory class for creating different fusion network
models. The BaseNetwork class is an abstract base class that defines the interface for a network model.

It has an abstract __init__ method and a forward method that needs to be implemented by subclasses.
The CNNFusionNetWrapper, EfficientNetSelfAttentionWrapper, and EfficientNetV2MultiHeadAttentionWrapper classes are
concrete implementations of the BaseNetwork interface. They wrap specific fusion network models (CNNFusionNet,
EfficientNetSelfAttention, and EfficientNetV2MultiHeadAttention, respectively) and provide a forward method that calls
the corresponding model's forward method.

The NetworkFactory class is responsible for creating the appropriate fusion network model based on the given fusion
network type. It has a static method create_network that takes the fusion network type, network configuration
parameters, and an optional device argument. It checks the fusion network type and creates the corresponding wrapper
class instance. It also sets the model's device to the specified device or the default device.
"""

import torch

from abc import ABC, abstractmethod
from typing import Optional

from fusion_models.cnn_fusion_net import CNNFusionNet
from fusion_models.efficient_net_self_attention import EfficientNetSelfAttention
from fusion_models.efficient_net_v2_s_multihead_attention import EfficientNetV2MultiHeadAttention


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
# +++++++++++++++++++++++++++++++++++++ C N N   F U S I O N N E T   W R A P P E R ++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CNNFusionNetWrapper(BaseNetwork):
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture):
        self.model = (
            CNNFusionNet(type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture))

    def forward(self, x):
        return self.model(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++ E F F I C I E N T N E T   S E L F   A T T E N T I O N   W R A P P E R ++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EfficientNetSelfAttentionWrapper(BaseNetwork):
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture):
        self.model = (
            EfficientNetSelfAttention(type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb,
                                      network_cfg_texture))

    def forward(self, x):
        return self.model(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++ E F F I C I E N T N E T   M U L T I   H E A D   A T T E N T I O N   W R A P P E R ++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EfficientNetV2MultiHeadAttentionWrapper(BaseNetwork):
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture):
        self.model = (
            EfficientNetV2MultiHeadAttention(type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb,
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

        if fusion_network_type == "CNNFusionNet":
            model = CNNFusionNetWrapper(type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb,
                                        network_cfg_texture).model
        elif fusion_network_type == "EfficientNetSelfAttention":
            model = EfficientNetSelfAttentionWrapper(type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb,
                                                     network_cfg_texture).model
        elif fusion_network_type == "EfficientNetV2SelfAttention":
            model = EfficientNetSelfAttentionWrapper(type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb,
                                                     network_cfg_texture).model
        elif fusion_network_type == "EfficientNetV2MultiHeadAttention":
            model = EfficientNetV2MultiHeadAttentionWrapper(type_of_net, network_cfg_contour, network_cfg_lbp,
                                                            network_cfg_rgb, network_cfg_texture).model
        else:
            raise ValueError("Wrong type was given!")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        return model
