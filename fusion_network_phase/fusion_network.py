import torch
import torch.nn as nn

from stream_network_phase.CNN import CNN


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ F U S I O N   N E T +++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class FusionNet(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- __ I N I T __ --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        """
        This is the __init__ function of the FusionNet.

        :return: None
        """

        super(FusionNet, self).__init__()
        list_of_channels_con_tex = [1, 32, 48, 64, 128, 192, 256]
        list_of_channels_rgb = [3, 64, 96, 128, 256, 384, 512]
        self.contour_network = CNN(list_of_channels_con_tex)
        self.rgb_network = CNN(list_of_channels_rgb)
        self.texture_network = CNN(list_of_channels_con_tex)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------- F O R W A R D --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x1, x2, x3) -> torch.Tensor:
        """
        This is the forward function of the FusionNet.

        :param x1: input tensor for contour stream, with shape [batch_size, 1, height, width]
        :param x2: input tensor for RGB stream, with shape [batch_size, 3, height, width]
        :param x3: input tensor for texture stream, with shape [batch_size, 1, height, width]
        :return: output tensor with shape [batch_size, 512] after passing through fully connected layers.
        """

        x1 = self.contour_network(x1)
        x2 = self.rgb_network(x2)
        x3 = self.texture_network(x3)
        x = torch.cat((x1, x2, x3,), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
