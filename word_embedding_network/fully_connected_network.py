import torch.nn as nn


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim-20)
        self.fc3 = nn.Linear(hidden_dim-20, hidden_dim-40)
        self.fc4 = nn.Linear(hidden_dim-40, output_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        return x


from torchsummary import summary
model = FullyConnectedNetwork(61, 100, 5).to("cuda")
model.train()
summary(model, (1, 61))