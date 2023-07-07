import torch

from abc import ABC, abstractmethod

from models.efficient_net_b0 import EfficientNet
from models.efficient_net_v2_b0 import EfficientNetV2
from models.efficient_net_b0_self_attention import EfficientNetSelfAttention
from models.CNN import CNN


class BaseNetwork(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass


class EfficientNetWrapper(BaseNetwork):
    def __init__(self, network_cfg):
        self.model = EfficientNet(loc=network_cfg.get('channels'), grayscale=network_cfg.get('grayscale'))

    def forward(self, x):
        return self.model(x)


class EfficientNetV2Wrapper(BaseNetwork):
    def __init__(self, network_cfg):
        self.model = EfficientNetV2(loc=network_cfg.get("channels"), grayscale=network_cfg.get('grayscale'))

    def forward(self, x):
        return self.model(x)


class EfficientNetSelfAttentionWrapper(BaseNetwork):
    def __init__(self, network_cfg):
        self.model = EfficientNetSelfAttention(loc=network_cfg.get('channels'), grayscale=network_cfg.get('grayscale'))

    def forward(self, x):
        return self.model(x)


class StreamNetworkWrapper(BaseNetwork):
    def __init__(self, network_cfg):
        self.model = CNN(network_cfg.get('channels'))

    def forward(self, x):
        return self.model(x)


class NetworkFactory:
    @staticmethod
    def create_network(network_type, network_cfg, device=None):
        if network_type == "EfficientNet":
            model = EfficientNetWrapper(network_cfg).model
        elif network_type == "EfficientNetV2":
            model = EfficientNetV2Wrapper(network_cfg).model
        elif network_type == "EfficientNetSelfAttention":
            model = EfficientNetSelfAttentionWrapper(network_cfg).model
        elif network_type == "CNN":
            model = StreamNetworkWrapper(network_cfg).model
        else:
            raise ValueError("Wrong type was given!")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        return model
