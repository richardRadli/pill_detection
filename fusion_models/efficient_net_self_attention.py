"""
File: efficient_net_self_attention.py
Author: Richárd Rádli
E-mail: radli.richard@mik.uni-pannon.hu
Date: Apr 12, 2023

Description: The program implements the EfficientNet with Fusion Net.
"""

import torch
import torch.nn as nn

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

        latest_con_pt_file = find_latest_file_in_latest_directory(network_cfg_contour.get("model_weights_dir"))
        latest_lbp_pt_file = find_latest_file_in_latest_directory(network_cfg_lbp.get("model_weights_dir"))
        latest_rgb_pt_file = find_latest_file_in_latest_directory(network_cfg_rgb.get("model_weights_dir"))
        latest_tex_pt_file = find_latest_file_in_latest_directory(network_cfg_texture.get("model_weights_dir"))

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

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- F O R W A R D --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        """
        This is the forward function of the FusionNet.

        :param x1: input tensor for contour stream, with shape [batch_size, 1, height, width]
        :param x2: input tensor for RGB stream, with shape [batch_size, 3, height, width]
        :param x3: input tensor for texture stream, with shape [batch_size, 1, height, width]
        :param x4: input tensor for LBP stream, with shape [batch_size, 1, height, width]

        :return: output tensor with shape [batch_size, 640] after passing through fully connected layer.
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

        return x

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------- S E L F   A T T E N T I O N ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention mechanism to the input tensor.

        :param x: Input tensor with shape [batch_size, embedded_dim].
        :return: Output tensor after applying self-attention with shape [batch_size, embedded_dim].
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

        :param x: Input tensor with shape [batch_size, embedded_dim].
        :return: Output tensor after applying self-attention with shape [batch_size, embedded_dim].
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

        :param keys: Tensor with shape [batch_size, embedded_dim, 1].
        :param values: Tensor with shape [batch_size, 1, embedded_dim].
        :param queries: Tensor with shape [batch_size, 1, embedded_dim].
        :return: Output tensor after applying self-attention with shape [batch_size, embedded_dim].
        """

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)

        return attended_values
