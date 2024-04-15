"""
File: efficient_net_self_attention.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program implements the EfficientNet with Fusion Net.
"""

import torch
import torch.nn as nn

from config.config import ConfigStreamNetwork
from stream_network_models.stream_network_selector import NetworkFactory
from utils.utils import find_latest_file_in_latest_directory


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++ E F F I C I E N T N E T S E L F A T T E N T I O N +++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class EfficientNetSelfAttention(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- __ I N I T __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, type_of_net: str, network_cfg_contour: dict, network_cfg_lbp: dict,
                 network_cfg_rgb: dict, network_cfg_texture: dict) -> None:
        """
        This is the initialization function of the EfficientNetSelfAttention class.

        :param type_of_net: Type of network to create.
        :param network_cfg_contour: Configuration for the contour network.
        :param network_cfg_lbp: Configuration for the LBP network.
        :param network_cfg_rgb: Configuration for the RGB network.
        :param network_cfg_texture: Configuration for the texture network.
        """

        super(EfficientNetSelfAttention, self).__init__()

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

        self.input_dim = (network_cfg_contour.get("embedded_dim") +
                          network_cfg_lbp.get("embedded_dim") +
                          network_cfg_rgb.get("embedded_dim") +
                          network_cfg_texture.get("embedded_dim"))

        assert (network_cfg_contour.get("embedded_dim") ==
                network_cfg_lbp.get("embedded_dim") ==
                network_cfg_texture.get("embedded_dim"))
        con_lbp_tex_dimension = network_cfg_contour.get("embedded_dim")
        rgb_dimension = network_cfg_rgb.get("embedded_dim")

        self.query = nn.Linear(con_lbp_tex_dimension, con_lbp_tex_dimension)
        self.key = nn.Linear(con_lbp_tex_dimension, con_lbp_tex_dimension)
        self.value = nn.Linear(con_lbp_tex_dimension, con_lbp_tex_dimension)

        self.query_rgb = nn.Linear(rgb_dimension, rgb_dimension)
        self.key_rgb = nn.Linear(rgb_dimension, rgb_dimension)
        self.value_rgb = nn.Linear(rgb_dimension, rgb_dimension)

        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.relu = nn.ReLU()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- F O R W A R D --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, contour_tensor: torch.Tensor, lbp_tensor: torch.Tensor, rgb_tensor: torch.Tensor,
                texture_tensor: torch.Tensor) -> torch.Tensor:
        """
        This is the forward function of the FusionNet.

        Args:
            contour_tensor: input tensor for contour stream, with shape [batch_size, 1, height, width]
            lbp_tensor: input tensor for LBP stream, with shape [batch_size, 1, height, width]
            rgb_tensor: input tensor for RGB stream, with shape [batch_size, 3, height, width]
            texture_tensor: input tensor for texture stream, with shape [batch_size, 1, height, width]

        Returns:
             Output tensor with shape [batch_size, 640] after passing through fully connected layers.
        """

        contour_tensor = self.contour_network(contour_tensor)
        lbp_tensor = self.lbp_network(lbp_tensor)
        rgb_tensor = self.rgb_network(rgb_tensor)
        texture_tensor = self.texture_network(texture_tensor)

        contour_tensor = self.self_attention(contour_tensor)
        lbp_tensor = self.self_attention(lbp_tensor)
        rgb_tensor = self.self_attention_rgb(rgb_tensor)
        texture_tensor = self.self_attention(texture_tensor)

        concatenated = torch.cat(
            (contour_tensor, lbp_tensor, rgb_tensor, texture_tensor), dim=1
        )
        concatenated = self.fc1(concatenated)
        concatenated = self.fc2(concatenated)
        concatenated = self.relu(concatenated)

        return concatenated

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- S E L F   A T T E N T I O N ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention mechanism to the input tensor.

        Args:
            x: Input tensor with shape [batch_size, embedded_dim].

        Returns:
             Output tensor after applying self-attention with shape [batch_size, embedded_dim].
        """

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        return self.self_attention_module(keys, values, queries)

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- S E L F   A T T E N T I O N   R G B --------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def self_attention_rgb(self, x: torch.Tensor):
        """
        Apply self-attention mechanism to the input tensor.

        Args:
            x: Input tensor with shape [batch_size, embedded_dim].

        Returns:
             Output tensor after applying self-attention with shape [batch_size, embedded_dim].
        """

        queries = self.query_rgb(x)
        keys = self.key_rgb(x)
        values = self.value_rgb(x)

        return self.self_attention_module(keys, values, queries)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ S E L F   A T T E N T I O N   M O D U L E -----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def self_attention_module(keys: torch.Tensor, values: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention mechanism to the input tensors.

        Args:
            keys: Tensor with shape [batch_size, embedded_dim, 1].
            values: Tensor with shape [batch_size, 1, embedded_dim].
            queries: Tensor with shape [batch_size, 1, embedded_dim].

        Return:
             Output tensor after applying self-attention with shape [batch_size, embedded_dim].
        """

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)

        return attended_values
