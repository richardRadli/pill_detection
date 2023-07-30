import torch
import torch.nn as nn

from stream_network_models.stream_network_selector import NetworkFactory


class CNNFusionNet(nn.Module):
    def __init__(self, type_of_net, network_cfg_contour, network_cfg_lbp, network_cfg_rgb, network_cfg_texture) -> None:
        """
        This is the __init__ function of the FusionNet.

        :param type_of_net:
        :param network_cfg_contour:
        :param network_cfg_lbp:
        :param network_cfg_rgb:
        """
        super(CNNFusionNet, self).__init__()

        self.contour_network = NetworkFactory.create_network(type_of_net, network_cfg_contour)
        self.lbp_network = NetworkFactory.create_network(type_of_net, network_cfg_lbp)
        self.rgb_network = NetworkFactory.create_network(type_of_net, network_cfg_rgb)
        self.texture_network = NetworkFactory.create_network(type_of_net, network_cfg_texture)

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
    def forward(self, x1, x2, x3, x4) -> torch.Tensor:
        x1 = self.contour_network(x1)
        x2 = self.lbp_network(x2)
        x3 = self.rgb_network(x3)
        x4 = self.texture_network(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
