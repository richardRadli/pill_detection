import torch.nn as nn
import torchvision.models as models


class EfficientNet(nn.Module):
    def __init__(self, loc):
        super(EfficientNet, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.fc = nn.Linear(loc[6], loc[4])

    def forward(self, x):
        x = self.model(x)
        return x
