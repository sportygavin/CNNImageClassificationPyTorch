import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
test_data = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 4, shuffle = True, num_workers = 2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 4, shuffle = True, num_workers = 2)

image, label = train_data[0]
image.size()
torch.Size([3, 32, 32])

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.conv2d(3, 12, 5)