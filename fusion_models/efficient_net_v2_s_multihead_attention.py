import torch.nn as nn
import torchvision.models as models


class EfficientNetV2MultiHeadAttention(nn.Module):
    def __init__(self, loc, grayscale=True):
        super(EfficientNetV2MultiHeadAttention, self).__init__()
        self.loc = loc
        self.grayscale = grayscale
        self.model = models.efficientnet_b0(weights='DEFAULT') # self.build_model()
        if self.grayscale:
            self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)

        self.input_dim = 1000
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=4)

    def forward(self, x):
        if self.grayscale:
            x = x.expand(-1, 3, -1, -1)
        x = self.model(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)

        batch_size, _, height, width = x.size()
        x = x.view(batch_size, height * width, -1)
        x = x.permute(1, 0, 2)
        queries = x
        keys = x
        values = x

        attention_output, _ = self.multihead_attention(queries, keys, values)
        attention_output = attention_output.permute(1, 0, 2)

        return attention_output.squeeze(1)

    def build_model(self):
        model = models.efficientnet_b0(weights='DEFAULT')
        for params in model.parameters():
            params.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.loc[4])
        return model


# from torchsummary import summary
# enma = EfficientNetMultiheadAttention([1, 32, 48, 64, 128, 192, 256])
# enma = enma.to("cuda")
# summary(enma, (1, 224, 224))

