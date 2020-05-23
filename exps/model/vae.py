import torch
import torch.nn as nn
from torch.autograd import Variable


class VaeEncoder(nn.Module):
    def __init__(self, hidden_size=3):
        super(VaeEncoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 4, 2, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(True))

        self.fc_encode1 = nn.Linear(16 * 7 * 7, hidden_size)
        self.fc_encode2 = nn.Linear(16 * 7 * 7, hidden_size)
        self.fc_decode = nn.Linear(hidden_size, 16 * 7 * 7)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(16, 16, 4, 2, 1),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(16, 1, 4, 2, 1),
                                     nn.Sigmoid())

    def encoder(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # print(out)
        return self.fc_encode1(out.view(out.size(0), -1)), self.fc_encode2(out.view(out.size(0), -1))

    def sampler(self, mean, std):
        var = std.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = Variable(eps)
        return eps.mul(var).add_(mean)

    def decoder(self, x):
        out = self.fc_decode(x)
        out = self.deconv1(out.view(x.size(0), 16, 7, 7))
        out = self.deconv2(out)
        return out

    def forward(self, x):
        mean, std = self.encoder(x)
        code = self.sampler(mean, std)
        out = self.decoder(code)
        return out, code, mean, std


if __name__ == "__main__":
    model = VaeEncoder()
    inputs = torch.rand((1, 28, 28))
    inputs = inputs.unsqueeze(dim=0)
    output = model(inputs)
    print(output)
