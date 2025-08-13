import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1024 * 512, 128 * 256 * 128),
            nn.ReLU(),
            nn.Unflatten(1,(128, 256, 128)),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh() #Based on transform.Compose
        )
    def forward(self, input):
        img = self.main(input)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, 2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, 1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(256*256*128, 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        val = self.main(input)
        return val

class GAN:
    def __init__(self, lr):
        super(GAN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.loss = nn.BCELoss()

        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr)