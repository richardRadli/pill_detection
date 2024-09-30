"""
File: cnn_fusion_net.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 30, 2023

Description: The program implements the CNN Fusion Net.
"""

import torch
import torch.nn as nn

from config.json_config import json_config_selector
from config.networks_paths_selector import substream_paths
from typing import Tuple

from stream_network_models.stream_network_selector import StreamNetworkFactory
from utils.utils import find_latest_file_in_latest_directory, load_config_json


class CNNFusionNet(nn.Module):
    def __init__(self) -> None:
        """
        This is the initialization function of the FusionNet class.
        """

        super(CNNFusionNet, self).__init__()

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

        self.input_dim = (
            (
                    contour_substream_network_cfg.get("embedded_dim") +
                    lbp_substream_network_cfg.get("embedded_dim") +
                    rgb_substream_network_cfg.get("embedded_dim") +
                    texture_substream_network_cfg.get("embedded_dim")
            )
        )

        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.bn = nn.BatchNorm1d(self.input_dim)
        self.dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(self.input_dim, self.input_dim)

    @staticmethod
    def freeze_networks(network):
        """

        Args:
            network:

        Returns:

        """

        for param in network.parameters():
            param.requires_grad = False

    def forward(self, x_contour: torch.Tensor, x_lbp: torch.Tensor, x_rgb: torch.Tensor, x_texture: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the FusionNet for both query and reference images.

        Args:
            x_contour: Input tensor for the contour network (query).
            x_lbp: Input tensor for the LBP network (query).
            x_rgb: Input tensor for the RGB network (query).
            x_texture: Input tensor for the texture network (query).

        Returns:
            Embedding for query and reference after fusion.
        """

        # Process query and reference images through each stream's metric learning model
        x_contour = self.network_con(x_contour)
        x_lbp = self.network_lbp(x_lbp)
        x_rgb = self.network_rgb(x_rgb)
        x_texture = self.network_tex(x_texture)

        # Concatenate embeddings for both query and reference
        fusion_out = torch.cat([x_contour, x_lbp, x_rgb, x_texture], dim=1)
       
        # Apply fully connected layers for query fusion
        out = self.fc1(fusion_out)
        out = self.bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
