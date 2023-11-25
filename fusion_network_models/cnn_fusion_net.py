"""
File: cnn_fusion_net.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Jul 30, 2023

Description: The program implements the CNN Fusion Net.
"""

import torch
import torch.nn as nn

from config.config import ConfigStreamNetwork
from stream_network_models.stream_network_selector import NetworkFactory
from utils.utils import find_latest_file_in_latest_directory


class CNNFusionNet(nn.Module):
    def __init__(self, type_of_net: str, network_cfg_contour: dict, network_cfg_lbp: dict,
                 network_cfg_rgb: dict, network_cfg_texture: dict) -> None:
        """
        This is the initialization function of the FusionNet class.

        :param type_of_net: Type of network to create.
        :param network_cfg_contour: Configuration for the contour network.
        :param network_cfg_lbp: Configuration for the LBP network.
        :param network_cfg_rgb: Configuration for the RGB network.
        :param network_cfg_texture: Configuration for the texture network.
        """

        super(CNNFusionNet, self).__init__()

        stream_net_cfg = ConfigStreamNetwork().parse()

        latest_con_pt_file = find_latest_file_in_latest_directory(
            path=network_cfg_contour.get("model_weights_dir").get("CNN").get(stream_net_cfg.dataset_type),
            type_of_loss=stream_net_cfg.type_of_loss_func
        )
        latest_lbp_pt_file = find_latest_file_in_latest_directory(
            path=network_cfg_lbp.get("model_weights_dir").get("CNN").get(stream_net_cfg.dataset_type),
            type_of_loss=stream_net_cfg.type_of_loss_func
        )
        latest_rgb_pt_file = find_latest_file_in_latest_directory(
            path=network_cfg_rgb.get("model_weights_dir").get("CNN").get(stream_net_cfg.dataset_type),
            type_of_loss=stream_net_cfg.type_of_loss_func
        )
        latest_tex_pt_file = find_latest_file_in_latest_directory(
            path=network_cfg_texture.get("model_weights_dir").get("CNN").get(stream_net_cfg.dataset_type),
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

        self.input_dim = (network_cfg_contour.get("channels")[4] +
                          network_cfg_lbp.get("channels")[4] +
                          network_cfg_rgb.get("channels")[4] +
                          network_cfg_texture.get("channels")[4])

        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- F O R W A R D --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FusionNet.

        :param x1: Input tensor for the contour network.
        :param x2: Input tensor for the LBP network.
        :param x3: Input tensor for the RGB network.
        :param x4: Input tensor for the texture network.
        :return: Output tensor after fusion and fully connected layers.
        """

        x1 = self.contour_network(x1)
        x2 = self.lbp_network(x2)
        x3 = self.rgb_network(x3)
        x4 = self.texture_network(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
