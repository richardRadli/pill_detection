import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetV2SelfAttention(nn.Module):
    def __init__(self, loc: list[int], grayscale=True):
        """
        EfficientNetV2 model with custom linear layer.

        :param loc: List of integers representing the number of channels at each layer.
        :param grayscale: Whether the input is grayscale or not. Defaults to True.
        """

        super(EfficientNetV2SelfAttention, self).__init__()
        self.loc = loc
        self.grayscale = grayscale
        self.model = models.efficientnet_v2_s(weights='DEFAULT') # self.build_model()
        if self.grayscale:
            self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.linear = nn.Linear(1000, self.loc[4])

        self.input_dim = loc[4]
        self.query = nn.Linear(self.input_dim, self.input_dim)
        self.key = nn.Linear(self.input_dim, self.input_dim)
        self.value = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EfficientNet model.

        :param x: Input tensor.
        :return: Output tensor.
        """

        if self.grayscale:
            x = x.expand(-1, 3, -1, -1)
        x = self.model(x)
        x = self.linear(x)

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        keys = keys.unsqueeze(2)
        values = values.unsqueeze(1)
        queries = queries.unsqueeze(1)

        result = torch.bmm(queries, keys)
        scores = result / (self.input_dim ** 0.5)
        attention = torch.softmax(scores, dim=2)
        weighted = torch.bmm(attention, values)

        return weighted.squeeze(1)

    def build_model(self) -> nn.Module:
        """
        Build the EfficientNetV2 model with a custom linear layer.

        :return: EfficientNetV2 model with custom linear layer.
        """

        model = models.efficientnet_v2_s(weights='DEFAULT')
        for params in model.parameters():
            params.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.loc[4])
        return model
