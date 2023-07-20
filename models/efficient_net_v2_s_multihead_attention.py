import torch
import torch.nn as nn
import torchvision.models as models

from typing import List


class EfficientNetV2MultiHeadAttention(nn.Module):
    def __init__(self, loc: List[int], num_heads: int = 4, grayscale: str = True):
        """
        The constructor initializes the model's parameters and sets up the base model, which is EfficientNet-V2-s in
        this case.
        :param loc: shape information of the output tensor
        :param num_heads: determines the number of attention heads used in the multi-head attention mechanism
        :param grayscale:  boolean flag to indicate whether the input images are grayscale
        """

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

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- F O R W A R D --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        The forward method defines the computation that takes place during the forward pass through the model.
        The input tensor x is processed through the EfficientNet-V2-s model.
        :param x: input tensor
        :return:
        """

        if self.grayscale:
            x = x.expand(-1, 3, -1, -1)
        """
        The input tensor x is passed through the EfficientNet-V2-s model.  
        The model extracts features from the input image.
        """
        x = self.model(x)
        """
        After the EfficientNet model, the code applies a linear layer (self.linear) to transform the extracted features 
        to the desired output size specified in loc[4].
        """
        x = self.linear(x)

        # Calculate the multi-head attention
        """
        The code then proceeds to perform multi-head attention. It iterates over num_heads and creates separate Linear 
        layers for each attention head for queries, keys, and values. 
        The attention heads are used to capture different relationships between the features.
        The calculations for queries, keys, and values are performed similarly to the self-attention mechanism. 
        The torch.bmm function is used for batch matrix multiplication to compute the attention scores, 
        followed by a softmax to get the attention weights.
        """
        multi_head_outputs = []
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
            multi_head_outputs.append(weighted.squeeze(1))

        # Concatenate the multi-head outputs along the feature dimension
        """
        The outputs from all the attention heads are concatenated along the feature dimension using torch.cat. 
        The concatenated tensor is then passed through a final linear layer (self.final_linear) to merge the multi-head 
        attention outputs into a final output tensor.
        """
        combined = torch.cat(multi_head_outputs, dim=1)

        # Apply the final linear layer to merge multi-head attention outputs
        """
        The final output tensor is returned, representing the result of EfficientNet with multi-head attention on the 
        input data.
        """
        final_output = self.final_linear(combined)

        return final_output

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- B U I L D   M O D E L ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def build_model(self):
        """
        This method is responsible for creating the EfficientNet-V2-s model and modifying it for the specific task.
        It replaces the last fully connected layer (classifier) of the EfficientNet model with a new linear layer to
        match the desired output size specified in loc[4].
        """

        model = models.efficientnet_v2_s(weights='DEFAULT')
        for params in model.parameters():
            params.requires_grad = False

        model.classifier[1] = nn.Linear(in_features=1280, out_features=self.loc[4])
        return model

# from torchsummary import summary
# model = EfficientNetV2MultiHeadAttention([1, 32, 48, 64, 128, 192, 256])
# model = model.to("cuda")
# summary(model, (1, 224, 224))