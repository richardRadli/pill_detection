"""
File: efficientnet.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: May 06, 2023

Description: The program implements the EfficientNet b0 with custom linear layer.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNet(nn.Module):
    def __init__(self, num_out_feature: int, grayscale: bool) -> None:
        """
        EfficientNet model with custom linear layer.

        Args:
            num_out_feature: Number of output features.
            grayscale: Whether the input is grayscale or not. Defaults to True.

        Returns:
            None
        """

        super(EfficientNet, self).__init__()
        self.num_out_feature = num_out_feature
        self.grayscale = grayscale
        self.model = self.build_model()
        if self.grayscale:
            self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EfficientNet model.

        Args:
            x: Input tensor.
        Returns:
             Output tensor.
        """

        if self.grayscale:
            x = x.expand(-1, 3, -1, -1)
        x = self.model(x)
        return x

    def build_model(self) -> nn.Module:
        """
        Build the EfficientNet V2 s model with a custom linear layer.

        Returns:
             EfficientNet V2 s model with custom linear layer.
        """

        model = models.efficientnet_v2_s(weights="DEFAULT")
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features,
                                        out_features=self.num_out_feature)
        return model
