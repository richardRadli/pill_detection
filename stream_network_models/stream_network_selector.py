"""
File: stream_network_selector.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 07, 2023

Description: The program implements the EfficientNet V2 s with custom linear layer.
"""

import torch

from abc import ABC, abstractmethod

from stream_network_models.efficientnet_v2s import EfficientNet


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++ B A S E   N E T W O R K ++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class BaseNetwork(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++ E F F I C I E N T N E T   W R A P P E R +++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EfficientNetWrapper(BaseNetwork):
    def __init__(self, network_cfg):
        self.model = (
            EfficientNet(
                num_out_feature=network_cfg.get('embedded_dim'),
                grayscale=network_cfg.get('grayscale')
            )
        )

    def forward(self, x):
        return self.model(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++ N E T   F A C T O R Y +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NetworkFactory:
    @staticmethod
    def create_network(network_type, network_cfg, device=None):
        if network_type == "EfficientNet":
            model = EfficientNetWrapper(network_cfg).model
        else:
            raise ValueError("Wrong type was given!")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        return model
