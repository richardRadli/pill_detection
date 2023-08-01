import torch
import torch.nn as nn

from stream_network_models.stream_network_selector import NetworkFactory


class EfficientNetV2MultiHeadAttention(nn.Module):
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture) -> None:
        """

        :param type_of_net:
        :param network_cfg_contour:
        :param network_cfg_lbp:
        :param network_cfg_rgb:
        :param network_cfg_texture:
        """

        super(EfficientNetV2MultiHeadAttention, self).__init__()

        self.contour_network = NetworkFactory.create_network(type_of_net, network_cfg_contour)
        self.lbp_network = NetworkFactory.create_network(type_of_net, network_cfg_lbp)
        self.rgb_network = NetworkFactory.create_network(type_of_net, network_cfg_rgb)
        self.texture_network = NetworkFactory.create_network(type_of_net, network_cfg_texture)

        self.multi_head_con = nn.MultiheadAttention(embed_dim=network_cfg_contour.get("embedded_dim"), num_heads=4)
        self.multi_head_lbp = nn.MultiheadAttention(embed_dim=network_cfg_lbp.get("embedded_dim"), num_heads=4)
        self.multi_head_rgb = nn.MultiheadAttention(embed_dim=network_cfg_rgb.get("embedded_dim"), num_heads=4)
        self.multi_head_tex = nn.MultiheadAttention(embed_dim=network_cfg_texture.get("embedded_dim"), num_heads=4)

        self.multi_head_modules = {
            "contour": self.multi_head_con,
            "lbp": self.multi_head_lbp,
            "rgb": self.multi_head_rgb,
            "texture": self.multi_head_tex
        }

        input_dim = (network_cfg_contour.get("embedded_dim") +
                     network_cfg_lbp.get("embedded_dim") +
                     network_cfg_rgb.get("embedded_dim") +
                     network_cfg_texture.get("embedded_dim"))

        self.fc1 = nn.Linear(input_dim, input_dim)

    def forward(self, x1, x2, x3, x4):
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

        x1 = self.multi_head_attention(x1, sub_stream="contour")
        x2 = self.multi_head_attention(x2, sub_stream="lbp")
        x3 = self.multi_head_attention(x3, sub_stream="rgb")
        x4 = self.multi_head_attention(x4, sub_stream="texture")

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fc1(x)

        return x

    def multi_head_attention(self, x, sub_stream: str):
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)

        batch_size, _, height, width = x.size()
        x = x.view(batch_size, height * width, -1)
        x = x.permute(1, 0, 2)
        queries = x
        keys = x
        values = x

        multi_head_module = self.multi_head_modules.get(sub_stream)
        if multi_head_module is None:
            raise ValueError("Invalid sub_stream value. I")

        attention_output, _ = multi_head_module(queries, keys, values)

        attention_output = attention_output.permute(1, 0, 2)
        return attention_output.squeeze(1)
