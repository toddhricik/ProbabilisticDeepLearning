import residualBlocks as rb
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from IPython.display import display


# Define a simple model that incorporates the residual block as done by Ho et al. 2020 DPPM paper
class model(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize a conv block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding='same') 
        # Initialize a residual block
        self.resBlock = rb.residualBlock_PixelCNN(inputChannelCount=256, outputChannelCount=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.resBlock(x)
        return x
    

if __name__ == "__main__":
    # Define some transformations
    
    transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.5, 0.5, 0.5), (0.4, 0.5,0.5))])

    # Download and load the CIFAR-10 training data
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Create a dataloader for the training set
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=1)

    # Initialize a model
    m = model()

    # Training loop
    for i, data  in enumerate(train_loader):
        x, y = data     
        out = m(x)
        break
