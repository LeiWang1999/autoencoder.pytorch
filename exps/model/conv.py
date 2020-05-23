import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # 1,28,28
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            # 16,10,10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # 16,5,5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            # 8,3,3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1)
            # 8,2,2
        )
        self.decoder = nn.Sequential(
            # 8,3,3
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            # 16, 5, 5
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            # 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            # 1, 28, 28
            nn.Tanh()  # 将输出值映射到-1~1之间
        )

    def forward(self, X):
        encode = self.encoder(X)
        decode = self.decoder(encode)
        return encode, decode


if __name__ == "__main__":
    model = ConvEncoder()
    inputs = torch.rand((1, 28, 28))
    inputs = inputs.unsqueeze(dim=0)
    output = model(inputs)
    print(output.size())
