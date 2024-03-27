import torch.nn as nn


class FullyConnectedNetwork(nn.Module):
    def __init__(self, neurons):
        super(FullyConnectedNetwork, self).__init__()

        self.fc1 = nn.Linear(neurons[0], neurons[1])
        self.fc2 = nn.Linear(neurons[1], neurons[2])
        self.fc3 = nn.Linear(neurons[2], neurons[3])
        self.fc4 = nn.Linear(neurons[3], neurons[4])

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
