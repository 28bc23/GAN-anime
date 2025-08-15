import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from PIL import Image

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

class ConvTransBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, output_padding):
        super(ConvTransBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.block(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, output_padding, leak):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(leak, True)
        )
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

class WGAN():
    def __init__(self, batch_size, use_cuda=True, lr_g = 2e-4, lr_d = 2e-4):
        super(WGAN, self).__init__()

        self.batch_size = batch_size

        if use_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.RandomHorizontalFlip(.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])



    def get_batch(self):
        idx = random.randint(0, 9)

        nums = random.sample(range(1000), self.batch_size)
        batch_files = [f"./data/000{idx}/000{n:03d}.png" for n in nums]

        batch = []
        for f in batch_files:
            img = Image.open(f).convert('RGB')
            tensor = self.transform(img)
            batch.append(tensor)
        batch = torch.stack(batch).to(self.device)
        return batch