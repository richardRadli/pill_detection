import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetV2(nn.Module):
    def __init__(self, loc: list[int], grayscale=True):
        """
        EfficientNetV2 model with custom linear layer.

        :param loc: List of integers representing the number of channels at each layer.
        :param grayscale: Whether the input is grayscale or not. Defaults to True.
        """

        super(EfficientNetV2, self).__init__()
        self.loc = loc
        self.grayscale = grayscale
        self.model = self.build_model()
        if self.grayscale:
            self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.model.fc = nn.Linear(self.loc[6], self.loc[4])

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

        model = models.efficientnet_v2_m(weights='DEFAULT')
        for params in model.parameters():
            params.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.loc[4])
        return model
