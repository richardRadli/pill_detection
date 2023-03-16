import torch
import torch.nn as nn

from stream_network_phase.stream_network import StreamNetwork


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        list_of_channels_con_tex = [1, 32, 48, 64, 128, 192, 256]
        list_of_channels_rgb = [3, 64, 96, 128, 256, 384, 512]
        self.contour_network = StreamNetwork(list_of_channels_con_tex)
        self.rgb_network = StreamNetwork(list_of_channels_rgb)
        self.texture_network = StreamNetwork(list_of_channels_con_tex)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

    def forward(self, x1, x2, x3):
        x1 = self.contour_network(x1)
        x2 = self.rgb_network(x2)
        x3 = self.texture_network(x3)
        x = torch.cat((x1, x2, x3,), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
