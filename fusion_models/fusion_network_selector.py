import torch

from abc import ABC, abstractmethod

from fusion_models.efficient_net_self_attention import EfficientNetSelfAttention


class BaseNetwork(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def forward(self, x):
        pass


class EfficientNetSelfAttentionWrapper(BaseNetwork):
    def __init__(self):
        self.model = EfficientNetSelfAttention(type_of_net=,
                                               network_cfg_contour=,
                                               network_cfg_lbp=,
                                               network_cfg_rgb=,
                                               network_cfg_texture=)

    def forward(self, x):
        return self.model(x)


class NetworkFactory:
    @staticmethod
    def create_network(network_type, network_cfg, device=None):
        if network_type == "EfficientNetV1SelfAttention":
            model = EfficientNetSelfAttentionWrapper().model