"""
File: efficient_net_v2_s_multihead_attention.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 19, 2023

Description: The program implements the EfficientNetV2 small multi-head attention with Fusion Net.
"""

import torch
import torch.nn as nn

from config.json_config import json_config_selector
from config.networks_paths_selector import substream_paths
from fusion_network_models.multihead_attention import MultiHeadAttention
from stream_network_models.stream_network_selector import StreamNetworkFactory
from utils.utils import find_latest_file_in_latest_directory, load_config_json


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++ E F F I C I E N T N E T V 2 M U L T I H E A D A T T E N T I O N +++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EfficientNetV2MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        """
            This is the initialization function of the EfficientNetSelfAttention class.
            """

        super(EfficientNetV2MultiHeadAttention, self).__init__()

        stream_net_cfg = (
            load_config_json(
                json_schema_filename=json_config_selector("stream_net").get("schema"),
                json_filename=json_config_selector("stream_net").get("config")
            )
        )

        contour_substream_network_cfg = stream_net_cfg.get("streams").get("Contour")
        lbp_substream_network_cfg = stream_net_cfg.get("streams").get("LBP")
        rgb_substream_network_cfg = stream_net_cfg.get("streams").get("RGB")
        texture_substream_network_cfg = stream_net_cfg.get("streams").get("Texture")

        self.dataset_type = stream_net_cfg.get("dataset_type")
        self.network_type = stream_net_cfg.get("type_of_net")

        contour_weight_files_path = (
            substream_paths().get("Contour").get(self.dataset_type).get(self.network_type).get("model_weights_dir")
        )
        lbp_weight_files_path = (
            substream_paths().get("LBP").get(self.dataset_type).get(self.network_type).get("model_weights_dir")
        )
        rgb_weight_files_path = (
            substream_paths().get("RGB").get(self.dataset_type).get(self.network_type).get("model_weights_dir")
        )
        texture_weight_files_path = (
            substream_paths().get("Texture").get(self.dataset_type).get(self.network_type).get("model_weights_dir")
        )

        latest_con_pt_file = find_latest_file_in_latest_directory(
            path=contour_weight_files_path
        )
        latest_lbp_pt_file = find_latest_file_in_latest_directory(
            path=lbp_weight_files_path,
        )
        latest_rgb_pt_file = find_latest_file_in_latest_directory(
            path=rgb_weight_files_path,
        )
        latest_tex_pt_file = find_latest_file_in_latest_directory(
            path=texture_weight_files_path,
        )

        self.network_con = StreamNetworkFactory.create_network(self.network_type, contour_substream_network_cfg)
        self.network_lbp = StreamNetworkFactory.create_network(self.network_type, lbp_substream_network_cfg)
        self.network_rgb = StreamNetworkFactory.create_network(self.network_type, rgb_substream_network_cfg)
        self.network_tex = StreamNetworkFactory.create_network(self.network_type, texture_substream_network_cfg)

        self.network_con.load_state_dict(torch.load(latest_con_pt_file))
        self.network_lbp.load_state_dict(torch.load(latest_lbp_pt_file))
        self.network_rgb.load_state_dict(torch.load(latest_rgb_pt_file))
        self.network_tex.load_state_dict(torch.load(latest_tex_pt_file))

        self.freeze_networks(self.network_con)
        self.freeze_networks(self.network_lbp)
        self.freeze_networks(self.network_rgb)
        self.freeze_networks(self.network_tex)

        contour_dim = contour_substream_network_cfg.get("embedded_dim")
        lbp_dim = lbp_substream_network_cfg.get("embedded_dim")
        rgb_dim = rgb_substream_network_cfg.get("embedded_dim")
        texture_dim = texture_substream_network_cfg.get("embedded_dim")

        self.input_dim = (
                contour_dim + lbp_dim + rgb_dim + texture_dim
        )

        self.multihead_attention_contour = (
            MultiHeadAttention(
                input_dim=contour_dim,
                num_heads=4
            )
        )
        self.multihead_attention_lbp = (
            MultiHeadAttention(
                input_dim=lbp_dim,
                num_heads=4
            )
        )
        self.multihead_attention_rgb = (
            MultiHeadAttention(
                input_dim=rgb_dim,
                num_heads=4
            )
        )
        self.multihead_attention_texture = (
            MultiHeadAttention(
                input_dim=texture_dim,
                num_heads=4
            )
        )

        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.bn = nn.BatchNorm1d(self.input_dim)
        self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(self.input_dim, self.input_dim)

    @staticmethod
    def freeze_networks(network):
        """
            Freeze the parameters of a given network.

            Args:
                network: The network whose parameters should be frozen.
            """

        for param in network.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- F O R W A R D --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x_contour: torch.Tensor, x_lbp: torch.Tensor, x_rgb: torch.Tensor, x_texture: torch.Tensor) \
            -> torch.Tensor:
        """
            This is the forward function of the FusionNet.

            Args:
                x_contour: input tensor for contour stream, with shape [batch_size, 1, height, width]
                x_lbp: input tensor for RGB stream, with shape [batch_size, 3, height, width]
                x_rgb: input tensor for texture stream, with shape [batch_size, 1, height, width]
                x_texture: input tensor for LBP stream, with shape [batch_size, 1, height, width]

            Returns:
                output tensor with shape [batch_size, 640] after passing through fully connected layer.
            """

        x_contour = self.network_con(x_contour)
        x_lbp = self.network_lbp(x_lbp)
        x_rgb = self.network_rgb(x_rgb)
        x_texture = self.network_tex(x_texture)

        contour_attention = self.multihead_attention_contour(x_contour)
        lbp_attention = self.multihead_attention_lbp(x_lbp)
        rgb_attention = self.multihead_attention_rgb(x_rgb)
        texture_attention = self.multihead_attention_texture(x_texture)

        fusion_out = torch.cat([contour_attention, lbp_attention, rgb_attention, texture_attention], dim=1)

        x = self.fc1(fusion_out)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x
