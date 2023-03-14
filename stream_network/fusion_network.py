import torch.nn as nn

from stream_network import StreamNetwork


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        list_of_channels = [1, 32, 48, 64, 128, 192, 256]
        self.stream1 = StreamNetwork(list_of_channels)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x1):
        x1 = self.stream1(x1)
        x = self.fc1(x1)
        x = self.fc2(x)
        return x

fn = FusionNet()
print(fn)