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
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ConvTransBlock(in_channel=latent_dim, out_channel=512, kernel_size=4, stride=2, padding=1, output_padding=1),
            ConvTransBlock(in_channel=512, out_channel=256, kernel_size=4, stride=2, padding=1, output_padding=1),
            ConvTransBlock(in_channel=256, out_channel=128, kernel_size=4, stride=2, padding=1, output_padding=1),
            ConvTransBlock(in_channel=128, out_channel=64, kernel_size=4, stride=2, padding=1, output_padding=1),
            ConvTransBlock(in_channel=64, out_channel=32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(64, 3, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.Tanh()
        )
    def f(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            ConvBlock(in_channel=3, out_channel=32, kernel_size=4, stride=2, padding=1, output_padding=1, leak=0.2), # 128x64; 64x32
            ConvBlock(in_channel=32, out_channel=64, kernel_size=4, stride=2, padding=1, output_padding=1, leak=0.2), # 64x32; 32x16
            ConvBlock(in_channel=64, out_channel=128, kernel_size=4, stride=2, padding=1, output_padding=1, leak=0.2), # 32x16; 16x8
            ConvBlock(in_channel=128, out_channel=256, kernel_size=4, stride=2, padding=1, output_padding=1, leak=0.2), # 16x8; 8x4
            ConvBlock(in_channel=256, out_channel=512, kernel_size=4, stride=2, padding=1, output_padding=1, leak=0.2), # 8x4; 4x2
            nn.Conv2d(512, 1, kernel_size=(4, 2), stride=1, padding=0),  # 4x2; 1x1
        )
    def f(self, x):
        return self.main(x)

class WGAN():
    def __init__(self, batch_size, latent_dim, use_cuda=True, lr_g = 2e-4, lr_d = 2e-4,g_betas = (0.5, 0.999), d_betas = (0.5, 0.999), c_lambda = 10):
        super(WGAN, self).__init__()

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.c_lambda = c_lambda

        if use_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.generator = Generator(self.latent_dim)
        self.discriminator = Discriminator()

        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=g_betas)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=d_betas)

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

    def g_loss(self, d_gen_val):
        g_loss = -torch.mean(d_gen_val)
        return g_loss
    def d_loss(self, d_fake_val, d_real_val, gradient_penalty):
        d_loss = -torch.mean(d_fake_val)
        return d_loss