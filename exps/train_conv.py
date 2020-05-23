import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from model.conv import ConvEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lib.utils import Writer

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
model = ConvEncoder()
model.to(device)
epochs = 200
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Draw graph for Visualization
inputs = torch.rand((1, 28, 28))
inputs = inputs.unsqueeze(dim=0).to(device)
writer.add_graph(model, inputs)

# Train
running_loss = 0.0
for epoch in range(epochs):
    for index, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        encode, decode = model(images)
        loss = criterion(decode, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    writer.add_scaler('trainning loss', running_loss, epoch * len(dataloader))
    running_loss = 0.0
    if epoch % 5 == 0:
        tag = "epoch : " + str(epoch)
        writer.add_images(tag=tag, images=images, logits=decode)
    print("epoch: %d, loss : %.3f" % (epoch, loss))

writer.close()

# Save model paramaters
torch.save(model.state_dict(), '../model/conv_encoder.pth')
