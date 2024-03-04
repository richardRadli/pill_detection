"""
File: efficient_net_multihead_attention.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 19, 2023

Description: The program implements the fusion of EfficientNet V2 s substreams with multi-head attention into a
Fusion Net.
"""

import torch
import torch.nn as nn

from config.config import ConfigStreamNetwork, ConfigFusionNetwork
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
        fusion_net_cfg = ConfigFusionNetwork().parse()

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

        self.multi_head_con = nn.MultiheadAttention(
            embed_dim=network_cfg_contour.get("embedded_dim"),
            num_heads=fusion_net_cfg.num_heads
        )
        self.multi_head_lbp = nn.MultiheadAttention(
            embed_dim=network_cfg_lbp.get("embedded_dim"),
            num_heads=fusion_net_cfg.num_heads
        )
        self.multi_head_rgb = nn.MultiheadAttention(
            embed_dim=network_cfg_rgb.get("embedded_dim"),
            num_heads=fusion_net_cfg.num_heads
        )
        self.multi_head_tex = nn.MultiheadAttention(
            embed_dim=network_cfg_texture.get("embedded_dim"),
            num_heads=fusion_net_cfg.num_heads
        )

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
    def forward(self,
                contour_embedding: torch.Tensor,
                lbp_embedding: torch.Tensor,
                rgb_embedding: torch.Tensor,
                texture_embedding: torch.Tensor):
        """
        This is the forward function of the EfficientNetMultiHeadAttention.

        :param contour_embedding: input tensor for contour stream, with shape [batch_size, 1, height, width]
        :param lbp_embedding: input tensor for LBP stream, with shape [batch_size, 1, height, width]
        :param rgb_embedding: input tensor for RGB stream, with shape [batch_size, 3, height, width]
        :param texture_embedding: input tensor for texture stream, with shape [batch_size, 1, height, width]

        :return: Output tensor with shape [batch_size, sum(contour_embedding, lbp_embedding, rgb_embedding,
        texture_embedding)] after passing through two fully connected layers.
        """

        contour_embedding = self.contour_network(contour_embedding)
        lbp_embedding = self.lbp_network(lbp_embedding)
        rgb_embedding = self.rgb_network(rgb_embedding)
        texture_embedding = self.texture_network(texture_embedding)

        contour_embedding = self.multi_head_attention(contour_embedding, sub_stream="contour")
        lbp_embedding = self.multi_head_attention(lbp_embedding, sub_stream="lbp")
        rgb_embedding = self.multi_head_attention(rgb_embedding, sub_stream="rgb")
        texture_embedding = self.multi_head_attention(texture_embedding, sub_stream="texture")

        x = torch.cat((contour_embedding, lbp_embedding, rgb_embedding, texture_embedding), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)

        return x

    # ------------------------------------------------------------------------------------------------------------------
    # -------------------------------------- M U L T I H E A D A T T E N T I O N ---------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def multi_head_attention(self,
                             x: torch.Tensor,
                             sub_stream: str):
        """
        This function implements the multi-head attention

        :param x: Input tensor
        :param sub_stream: Type of sub stream
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
