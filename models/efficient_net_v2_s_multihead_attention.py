import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetV2MultiHeadAttention(nn.Module):
    def __init__(self, loc, num_heads=4, grayscale=True):
        super(EfficientNetV2MultiHeadAttention, self).__init__()
        self.loc = loc
        self.grayscale = grayscale
        self.model = models.efficientnet_v2_s(weights='DEFAULT')  # self.build_model()
        if self.grayscale:
            self.model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
        self.linear = nn.Linear(1000, loc[4])  # loc[6]

        self.input_dim = loc[4]
        self.num_heads = num_heads

        # Create separate Linear layers for each head
        self.queries = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(num_heads)])
        self.keys = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(num_heads)])
        self.values = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(num_heads)])

        # Final linear layer to merge multi-head attention outputs
        self.final_linear = nn.Linear(self.input_dim * num_heads, self.input_dim)

    def forward(self, x):
        if self.grayscale:
            x = x.expand(-1, 3, -1, -1)
        x = self.model(x)
        x = self.linear(x)

        # Calculate multi-head attention
        multihead_outputs = []
        for i in range(self.num_heads):
            queries = self.queries[i](x)
            keys = self.keys[i](x)
            values = self.values[i](x)

            keys = keys.unsqueeze(2)
            values = values.unsqueeze(1)
            queries = queries.unsqueeze(1)

            result = torch.bmm(queries, keys)
            scores = result / (self.input_dim ** 0.5)
            attention = torch.softmax(scores, dim=2)
            weighted = torch.bmm(attention, values)

            # Append the output from each head to the list
            multihead_outputs.append(weighted.squeeze(1))

        # Concatenate the multi-head outputs along the feature dimension
        combined = torch.cat(multihead_outputs, dim=1)

        # Apply the final linear layer to merge multi-head attention outputs
        final_output = self.final_linear(combined)

        return final_output

    def build_model(self):
        model = models.efficientnet_v2_s(weights='DEFAULT')
        for params in model.parameters():
            params.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.loc[4])
        return model
