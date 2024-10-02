"""
File: fusion_network_selector.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 26, 2023
"""

import torch

from abc import ABC, abstractmethod
from typing import Optional

from fusion_network_models.efficient_net_v2_self_attention import EfficientNetV2SelfAttention
from fusion_network_models.efficient_net_v2_s_multihead_attention import EfficientNetV2MultiHeadAttention
from fusion_network_models.efficient_net_v2_s_mha_fmha import EfficientNetV2MHAFMHA


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
class EfficientNetV2SelfAttentionWrapper(BaseNetwork):
    def __init__(self):
        self.model = (
            EfficientNetV2SelfAttention()
        )

    def forward(self, x):
        return self.model(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++ E F F I C I E N T N E T   M U L T I   H E A D   A T T E N T I O N   W R A P P E R ++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EfficientNetV2MultiHeadAttentionWrapper(BaseNetwork):
    def __init__(self):
        self.model = (
            EfficientNetV2MultiHeadAttention()
        )

    def forward(self, x):
        return self.model(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++ E F F I C I E N T N E T   M U L T I   H E A D   A T T E N T I O N   W R A P P E R ++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EfficientNetV2MHAFMHAWrapper(BaseNetwork):
    def __init__(self):
        self.model = (
            EfficientNetV2MHAFMHA()
        )

    def forward(self, x):
        return self.model(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++ N E T   F A C T O R Y +++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class FusionNetworkFactory:
    @staticmethod
    def create_network(fusion_network_type: str, device: Optional[torch.device] = None) \
            -> torch.nn.Module:
        """
        Create a fusion network model based on the given fusion network type.

        Args:
            fusion_network_type: The type of fusion network to create.
            device: The device to use for the model (default: GPU if available, otherwise CPU).

        Return:
             The created fusion network model.
        """

        if fusion_network_type == "EfficientNetV2SelfAttention":
            model = EfficientNetV2SelfAttentionWrapper().model
        elif fusion_network_type == "EfficientNetV2MultiHeadAttention":
            model = EfficientNetV2MultiHeadAttentionWrapper().model
        elif fusion_network_type == "EfficientNetV2MHAFMHA":
            model = EfficientNetV2MHAFMHAWrapper().model
        else:
            raise ValueError("Wrong type was given!")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        return model
