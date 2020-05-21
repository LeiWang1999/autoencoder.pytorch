import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from model import AutoEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
# Tensorboard Writer
writer = SummaryWriter('./runs/fashionmnist')

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
epochs = 20
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
iter(dataloader).next()
for epoch in range(epochs):
    for index, (images, labels) in enumerate(dataloader):
        logits = model(images)
        loss = criterion(logits, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch: %d, loss : %.3f" % (epoch, loss))
