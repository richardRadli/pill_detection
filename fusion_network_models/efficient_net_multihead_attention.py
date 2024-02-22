"""
File: efficient_net_multihead_attention.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 19, 2023

Description: The program implements the EfficientNet b0 multi-head attention with Fusion Net.
"""

import torch
import torch.nn as nn

from config.config import ConfigStreamNetwork
from stream_network_models.stream_network_selector import NetworkFactory
from utils.utils import find_latest_file_in_latest_directory


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++ E F F I C I E N T N E T V 2 M U L T I H E A D A T T E N T I O N +++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EfficientNetMultiHeadAttention(nn.Module):
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture) -> None:
        """
        This is the initialization function of the EfficientNetMultiHeadAttention class.

        :param type_of_net: Type of network to create.
        :param network_cfg_contour: Configuration for the contour network.
        :param network_cfg_lbp: Configuration for the LBP network.
        :param network_cfg_rgb: Configuration for the RGB network.
        :param network_cfg_texture: Configuration for the texture network.
        """

        super(EfficientNetMultiHeadAttention, self).__init__()

        stream_net_cfg = ConfigStreamNetwork().parse()

        latest_con_pt_file = find_latest_file_in_latest_directory(
            path=(
                network_cfg_contour
                .get("model_weights_dir")
                .get("EfficientNet")
                .get(stream_net_cfg.dataset_type)
            ),
            type_of_loss=stream_net_cfg.type_of_loss_func
        )
        latest_lbp_pt_file = find_latest_file_in_latest_directory(
            path=(
                network_cfg_lbp
                .get("model_weights_dir")
                .get("EfficientNet")
                .get(stream_net_cfg.dataset_type)
            ),
            type_of_loss=stream_net_cfg.type_of_loss_func
        )
        latest_rgb_pt_file = find_latest_file_in_latest_directory(
            path=(
                network_cfg_rgb
                .get("model_weights_dir")
                .get("EfficientNet")
                .get(stream_net_cfg.dataset_type)
            ),
            type_of_loss=stream_net_cfg.type_of_loss_func
        )
        latest_tex_pt_file = find_latest_file_in_latest_directory(
            path=(
                network_cfg_texture
                .get("model_weights_dir")
                .get("EfficientNet")
                .get(stream_net_cfg.dataset_type)
            ),
            type_of_loss=stream_net_cfg.type_of_loss_func
        )

        self.contour_network = NetworkFactory.create_network(type_of_net, network_cfg_contour)
        self.lbp_network = NetworkFactory.create_network(type_of_net, network_cfg_lbp)
        self.rgb_network = NetworkFactory.create_network(type_of_net, network_cfg_rgb)
        self.texture_network = NetworkFactory.create_network(type_of_net, network_cfg_texture)

        self.contour_network.load_state_dict(torch.load(latest_con_pt_file))
        self.lbp_network.load_state_dict(torch.load(latest_lbp_pt_file))
        self.rgb_network.load_state_dict(torch.load(latest_rgb_pt_file))
        self.texture_network.load_state_dict(torch.load(latest_tex_pt_file))

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
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- F O R W A R D --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor):
        """
        This is the forward function of the FusionNet.

        :param x1: input tensor for contour stream, with shape [batch_size, 1, height, width]
        :param x2: input tensor for LBP stream, with shape [batch_size, 1, height, width]
        :param x3: input tensor for RGB stream, with shape [batch_size, 3, height, width]
        :param x4: input tensor for texture stream, with shape [batch_size, 1, height, width]

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
        x = self.fc2(x)
        x = self.relu(x)

        return x

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------- M U L T I H E A D A T T E N T I O N ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def multi_head_attention(self, x: torch.Tensor, sub_stream: str):
        """
        This function implements the multi-head attention

        :param x: Input tensor
        :param sub_stream: type of sub stream
        :return:
        """

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
