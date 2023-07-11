import torch
import torch.nn as nn

from models.network_selector import NetworkFactory


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ F U S I O N   N E T +++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class FusionNet(nn.Module):
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

        super(FusionNet, self).__init__()

        self.contour_network = NetworkFactory.create_network(type_of_net, network_cfg_contour)
        self.lbp_network = NetworkFactory.create_network(type_of_net, network_cfg_lbp)
        self.rgb_network = NetworkFactory.create_network(type_of_net, network_cfg_rgb)
        self.texture_network = NetworkFactory.create_network(type_of_net, network_cfg_texture)
        self.fc1 = nn.Linear(640, 640)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(640, 640)

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
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
