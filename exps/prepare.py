import torch
import torchvision.datasets as datasets
import os

root = './data'
if os.path.isdir(root) and os.path.isdir(os.path.join(root, 'MNIST')):
    print("dataset alredy existed")
    print("You can Check it in ", root)
else:
    datasets.MNIST(root=root,
                   download=True)
