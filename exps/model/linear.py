import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    def __init__(self):
        super(LinearEncoder, self).__init__()
        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            # compress to 3 features which can be visualized in plt # 压缩成3个特征, 进行 3D 图像可视化
            nn.Linear(12, 3),
        )
        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Tanh(),       # compress to a range (0, 1) # 激励函数让输出值在 (0, 1)
        )

    def forward(self, X):
        X = X.view(-1, 28*28)
        encode = self.encoder(X)
        decode = self.decoder(encode)
        decode = decode.view(-1, 1, 28, 28)
        return encode, decode


if __name__ == "__main__":
    model = LinearEncoder()
    inputs = torch.rand((1, 28, 28))
    inputs = inputs.unsqueeze(dim=0)
    output = model(inputs)
    print(output)
