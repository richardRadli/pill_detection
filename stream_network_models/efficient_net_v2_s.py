"""
File: efficient_net_v2_s.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 07, 2023

Description: The program implements the EfficientNet V2 s with custom linear layer.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetV2Small(nn.Module):
    def __init__(self, num_out_feature: int = 128, grayscale=True):
        """
        EfficientNetV2 model with custom linear layer.

        :param num_out_feature: Number of output feature.
        :param grayscale: Whether the input is grayscale or not. Defaults to True.
        """

        super(EfficientNetV2Small, self).__init__()
        self.num_out_feature = num_out_feature
        self.grayscale = grayscale
        self.model = self.build_model()
        if self.grayscale:
            self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EfficientNet model.

        :param x: Input tensor.
        :return: Output tensor.
        """

        if self.grayscale:
            x = x.expand(-1, 3, -1, -1)
        x = self.model(x)
        return x

    def build_model(self) -> nn.Module:
        """
        Build the EfficientNetV2 model with a custom linear layer.

        :return: EfficientNetV2 model with custom linear layer.
        """

        model = models.efficientnet_v2_m(weights='IMAGENET1K_V1')
        for params in model.parameters():
            params.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.num_out_feature)

        return model
