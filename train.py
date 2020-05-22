import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from model import AutoEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import Writer
# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current device is ", device)

# Tensorboard Writer
writer = Writer(log_dir='./runs/mnist')

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
model = AutoEncoder()
model.to(device)
epochs = 200
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
for epoch in range(epochs):
    for index, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        logits = model(images)
        loss = criterion(logits, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        tag = "epoch : " + str(epoch)
        writer.add_images(tag=tag, images=images, logits=logits)
    print("epoch: %d, loss : %.3f" % (epoch, loss))
writer.close()
