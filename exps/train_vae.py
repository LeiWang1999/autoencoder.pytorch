import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from model.vae import VaeEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lib.utils import Writer


def loss_f(out, target, mean, std, bce):
    bceloss = bce(out, target)
    latent_loss = torch.sum(mean.pow(2).add_(
        std.exp()).mul_(-1).add_(1).add_(std)).mul_(-0.5)
    return bceloss + latent_loss


# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current device is ", device)

# Tensorboard Writer
writer = Writer(log_dir='./runs/vae')

# Load MNIST images
root = './data'
download = False
batch_size = 128
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.MNIST(root=root,
                         download=download, transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Initialze Model
model = VaeEncoder()
model.to(device)
epochs = 20
criterion = nn.BCELoss().to(device)
criterion.size_average = False
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Draw graph for Visualization
inputs = torch.rand((1, 28, 28))
inputs = inputs.unsqueeze(dim=0)
writer.add_graph(model, inputs)

# Train
running_loss = 0.0
for epoch in range(epochs):
    for index, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        output, _, mean, std = model(images)
        loss = loss_f(output, data, mean, std, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    writer.add_scaler('trainning vae loss', running_loss,
                      epoch * len(dataloader))
    running_loss = 0.0
    if epoch % 5 == 0:
        tag = "epoch : " + str(epoch)
        writer.add_images(tag=tag, images=images, logits=decode)
    print("epoch: %d, loss : %.3f" % (epoch, loss))

writer.close()

# Save model paramaters
torch.save(model.state_dict(), './model/vae_encoder.pth')
