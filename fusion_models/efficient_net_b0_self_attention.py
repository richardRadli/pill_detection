import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetSelfAttention(nn.Module):
    def __init__(self, loc, grayscale=True):
        super(EfficientNetSelfAttention, self).__init__()
        self.loc = loc
        self.grayscale = grayscale
        self.model = self.build_model()
        if self.grayscale:
            self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.linear = nn.Linear(1000, loc[4])  # loc[6]



    def forward(self, x):
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

    def build_model(self):
        model = models.efficientnet_b0(weights='DEFAULT')
        for params in model.parameters():
            params.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.loc[4])
        return model
