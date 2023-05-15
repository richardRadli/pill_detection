import torch.nn as nn
import torchvision.models as models


class EfficientNetSelfAttention(nn.Module):
    def __init__(self, loc, grayscale=True):
        super(EfficientNetSelfAttention, self).__init__()
        self.loc = loc
        self.grayscale = grayscale
        self.model = models.efficientnet_b0(weights='DEFAULT') # self.build_model()
        if self.grayscale:
            self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.linear = nn.Linear(1000, loc[4])
        self.self_attention = nn.MultiheadAttention(embed_dim=loc[4], num_heads=loc[4])
        self.fc = nn.Linear(loc[4], loc[4])

    def forward(self, x):
        if self.grayscale:
            x = x.expand(-1, 3, -1, -1)
        x = self.model(x)
        x = self.linear(x)
        x, _ = self.self_attention(x, x, x)  # Self-attention
        x = self.fc(x)  # Linear layer
        return x

    def build_model(self):
        model = models.efficientnet_b0(weights='DEFAULT')
        for params in model.parameters():
            params.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.loc[6])
        return model


from torchsummary import summary
en = EfficientNetSelfAttention(loc=[3, 64, 96, 128, 256, 384, 512])
en = en.to("cuda")
summary(en, (3, 128, 128))
