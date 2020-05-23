import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class VaeEncoder(nn.Module):
    def __init__(self, hidden_size=3):
        super().__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, hidden_size)  # fc21 for mean of Z
        self.fc22 = nn.Linear(500, hidden_size)  # fc22 for log variance of Z
        self.fc3 = nn.Linear(hidden_size, 500)
        self.fc4 = nn.Linear(500, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        # I guess the reason for using logvar instead of std or var is that
        # the output of fc22 can be negative value (std and var should be positive)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # x: [batch size, 1, 28,28] -> x: [batch size, 784]
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decode = self.decode(z)
        return decode, mu, logvar


if __name__ == "__main__":
    model = VaeEncoder()
    inputs = torch.rand((1, 28, 28))
    inputs = inputs.unsqueeze(dim=0)
    output = model(inputs)
    print(output)
