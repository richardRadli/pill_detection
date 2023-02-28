import torch
import torch.nn as nn


# Fuse the two networks
class FusionNet(nn.Module):
    def __init__(self, net1, net2, net3):
        super(FusionNet, self).__init__()

        self.net1 = net1
        self.net2 = net2
        self.net3 = net3

        # get output sizes of nets
        net1_out_size = 256
        net2_out_size = 128
        net3_out_size = 128

        # create linear layers for fusion
        self.fc1 = nn.Linear(net1_out_size, net2_out_size)
        self.fc2 = nn.Linear(net2_out_size, net3_out_size)

    def forward(self, x):
        # pass input through each network
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3 = self.net3(x2)

        # flatten output of net1
        x1 = x1.view(x1.size(0), -1)

        # pass net1 output through fc1
        x1 = self.fc1(x1)

        # pass net2 output through fc2
        x2 = self.fc2(x2)

        # concatenate net1, net2, and net3 outputs
        x = torch.cat((x1, x2, x3), dim=1)

        return x
