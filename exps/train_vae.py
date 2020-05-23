import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from model.vae import VaeEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lib.utils import Writer
from torch.nn import functional as F

# --- defines the loss function --- #


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

    return BCE + KLD


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
    transforms.ToTensor()
])
dataset = datasets.MNIST(root=root,
                         download=download, transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Initialze Model
model = VaeEncoder()
model.to(device)
epochs = 200
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Draw graph for Visualization
model.eval()
inputs = torch.rand((1, 28, 28))
inputs = inputs.unsqueeze(dim=0)
writer.add_graph(model, inputs.to(device))

# Train
running_loss = 0.0
model.train()
for epoch in range(epochs):
    for index, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        decode, mu, logvar = model(images)
        loss = loss_function(decode, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    writer.add_scaler('trainning vae loss', running_loss,
                      epoch * len(dataloader))
    running_loss = 0.0
    if epoch % 5 == 0:
        tag = "epoch : " + str(epoch)
        decode = decode.view(-1, 1, 28, 28)
        writer.add_images(tag=tag, images=images, logits=decode)
    print("epoch: %d, loss : %.3f" % (epoch, loss))

writer.close()

# Save model paramaters
torch.save(model.state_dict(), './model/vae_encoder.pth')
