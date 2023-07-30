import torch

from abc import ABC, abstractmethod

from fusion_models.cnn_fusion_net import CNNFusionNet
from fusion_models.efficient_net_self_attention import EfficientNetSelfAttention
from fusion_models.efficient_net_v2_s_multihead_attention import EfficientNetV2MultiHeadAttention


class BaseNetwork(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def forward(self, x):
        pass


class CNNFusionNetWrapper(BaseNetwork):
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture):
        self.model = (
            CNNFusionNet(type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture))

    def forward(self, x):
        return self.model(x)


class EfficientNetSelfAttentionWrapper(BaseNetwork):
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture):
        self.model = (
            EfficientNetSelfAttention(type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb,
                                      network_cfg_texture))

    def forward(self, x):
        return self.model(x)


class EfficientNetV2MultiHeadAttentionWrapper(BaseNetwork):
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture):
        self.model = (
            EfficientNetV2MultiHeadAttention(type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb,
                                             network_cfg_texture))

    def forward(self, x):
        return self.model(x)


class NetworkFactory:
    @staticmethod
    def create_network(fusion_network_type, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb,
                       network_cfg_texture, device=None):
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
