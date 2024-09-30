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
        for param in network.parameters():
            param.requires_grad = False

    def forward(self,
                x_contour_query: torch.Tensor, x_lbp_query: torch.Tensor, x_rgb_query: torch.Tensor, x_texture_query: torch.Tensor,
                x_contour_ref: torch.Tensor, x_lbp_ref: torch.Tensor, x_rgb_ref: torch.Tensor, x_texture_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the FusionNet for both query and reference images.

        Args:
            x_contour_query: Input tensor for the contour network (query).
            x_lbp_query: Input tensor for the LBP network (query).
            x_rgb_query: Input tensor for the RGB network (query).
            x_texture_query: Input tensor for the texture network (query).
            x_contour_ref: Input tensor for the contour network (reference).
            x_lbp_ref: Input tensor for the LBP network (reference).
            x_rgb_ref: Input tensor for the RGB network (reference).
            x_texture_ref: Input tensor for the texture network (reference).

        Returns:
            Embedding for query and reference after fusion.
        """

        # Process query and reference images through each stream's metric learning model
        x_contour_query, x_contour_ref = self.network_con(x_contour_query, x_contour_ref)
        x_lbp_query, x_lbp_ref = self.network_lbp(x_lbp_query, x_lbp_ref)
        x_rgb_query, x_rgb_ref = self.network_rgb(x_rgb_query, x_rgb_ref)
        x_texture_query, x_texture_ref = self.network_tex(x_texture_query, x_texture_ref)

        # Concatenate embeddings for both query and reference
        query_fusion_out = torch.cat([x_contour_query, x_lbp_query, x_rgb_query, x_texture_query], dim=1)
        ref_fusion_out = torch.cat([x_contour_ref, x_lbp_ref, x_rgb_ref, x_texture_ref], dim=1)

        # Apply fully connected layers for query fusion
        query_out = self.fc1(query_fusion_out)
        query_out = self.bn(query_out)
        query_out = torch.relu(query_out)
        query_out = self.dropout(query_out)
        query_out = self.fc2(query_out)

        # Apply fully connected layers for reference fusion
        ref_out = self.fc1(ref_fusion_out)
        ref_out = self.bn(ref_out)
        ref_out = torch.relu(ref_out)
        ref_out = self.dropout(ref_out)
        ref_out = self.fc2(ref_out)

        return query_out, ref_out

