import residualBlocks as rb
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from IPython.display import display

# Define some transformations
transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5, 0.5, 0.5), (0.4, 0.5,0.5))])

# Download and load the CIFAR-10 training data
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

# Create a dataloader for the training set
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

# Run 
resBlock = rb.residualBlock_PixelCNN(inputChannelCount=3, outputChannelCount=10)
out = resBlock(img)
