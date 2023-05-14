import torch.nn as nn
import torchvision.models as models


class EfficientNet(nn.Module):
    def __init__(self, loc, grayscale=True):
        super(EfficientNet, self).__init__()
        self.loc = loc
        self.grayscale = grayscale
        self.model = self.build_model()
        if self.grayscale:
            self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.model.fc = nn.Linear(self.loc[6], self.loc[4])

    def forward(self, x):
        if self.grayscale:
            x = x.expand(-1, 3, -1, -1)
        x = self.model(x)
        return x

    def build_model(self):
        model = models.efficientnet_b0(weights='DEFAULT')
        for params in model.parameters():
            params.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.loc[4])
        return model


# class EfficientNet(nn.Module):
#     def __init__(self, loc, grayscale=True):
#         super(EfficientNet, self).__init__()
#         self.grayscale = grayscale
#         self.model = models.efficientnet_b0(weights='DEFAULT')
#         if self.grayscale:
#             self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
#         self.linear = nn.Linear(1000, loc[6])
#         self.self_attention = nn.MultiheadAttention(embed_dim=loc[6], num_heads=loc[4])
#         self.fc = nn.Linear(loc[6], loc[4])
#
#     def forward(self, x):
#         if self.grayscale:
#             x = x.expand(-1, 3, -1, -1)
#         x = self.model(x)
#         x = self.linear(x)
#         x, _ = self.self_attention(x, x, x)  # Self-attention
#         x = self.fc(x)  # Linear layer
#         return x
