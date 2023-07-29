import torch
import torch.nn as nn

from stream_network_models.stream_network_selector import NetworkFactory


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ F U S I O N   N E T +++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EfficientNetSelfAttention(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- __ I N I T __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture) -> None:
        """
        This is the __init__ function of the FusionNet.

        :param type_of_net:
        :param network_cfg_contour:
        :param network_cfg_lbp:
        :param network_cfg_rgb:
        :param network_cfg_texture:
        """

        super(EfficientNetSelfAttention, self).__init__()

        self.contour_network = NetworkFactory.create_network(type_of_net, network_cfg_contour)
        self.lbp_network = NetworkFactory.create_network(type_of_net, network_cfg_lbp)
        self.rgb_network = NetworkFactory.create_network(type_of_net, network_cfg_rgb)
        self.texture_network = NetworkFactory.create_network(type_of_net, network_cfg_texture)
        self.fc1 = nn.Linear(640, 640)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(640, 640)

        self.input_dim = 640
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)

        self.query_rgb = nn.Linear(256, 256)
        self.key_rgb = nn.Linear(256, 256)
        self.value_rgb = nn.Linear(256, 256)

        self.multi_head_attention = nn.MultiheadAttention(embed_dim=640, num_heads=4)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- F O R W A R D --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x1, x2, x3, x4) -> torch.Tensor:
        """
        This is the forward function of the FusionNet.

        :param x1: input tensor for contour stream, with shape [batch_size, 1, height, width]
        :param x2: input tensor for RGB stream, with shape [batch_size, 3, height, width]
        :param x3: input tensor for texture stream, with shape [batch_size, 1, height, width]
        :param x4: input tensor for LBP stream, with shape [batch_size, 1, height, width]

        :return: output tensor with shape [batch_size, 640] after passing through fully connected layers.
        """

        x1 = self.contour_network(x1)
        x2 = self.lbp_network(x2)
        x3 = self.rgb_network(x3)
        x4 = self.texture_network(x4)

        x1 = self.self_attention(x1)
        x2 = self.self_attention(x2)
        x3 = self.self_attention_rgb(x3)
        x4 = self.self_attention(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def self_attention(self, x):
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

    def self_attention_rgb(self, x):
        queries = self.query_rgb(x)
        keys = self.key_rgb(x)
        values = self.value_rgb(x)

        keys = keys.unsqueeze(2)
        values = values.unsqueeze(1)
        queries = queries.unsqueeze(1)

        result = torch.bmm(queries, keys)
        scores = result / (self.input_dim ** 0.5)
        attention = torch.softmax(scores, dim=2)
        weighted = torch.bmm(attention, values)

        return weighted.squeeze(1)
