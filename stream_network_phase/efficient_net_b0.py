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


