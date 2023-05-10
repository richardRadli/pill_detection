import torch.nn as nn
import torchvision.models as models


class EfficientNet(nn.Module):
    def __init__(self, loc, grayscale=True):
        super(EfficientNet, self).__init__()
        self.grayscale = grayscale
        self.model = models.efficientnet_b0(weights='DEFAULT')
        if self.grayscale:
            self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.linear = nn.Linear(1000, loc[6])
        self.self_attention = nn.MultiheadAttention(embed_dim=loc[6], num_heads=8)
        self.fc = nn.Linear(loc[6], loc[4])

    def forward(self, x):
        if self.grayscale:
            x = x.expand(-1, 3, -1, -1)
        x = self.model(x)
        x = self.linear(x)
        x, _ = self.self_attention(x, x, x)  # Self-attention
        x = self.fc(x)  # Linear layer
        return x
