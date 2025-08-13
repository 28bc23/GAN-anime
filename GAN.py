import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from PIL import Image

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128 * 64 * 32),
            nn.ReLU(),
            nn.Unflatten(1,(128, 64, 32)),

            nn.Upsample(scale_factor=4),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
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

            nn.Conv2d(128, 128, 3, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, 1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(2097152, 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        val = self.main(input)
        return val

class GAN:
    def __init__(self, lr, latent_dim, batch_size, epochs):
        super(GAN, self).__init__()

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = Generator(self.latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.loss = nn.BCELoss()

        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.total_g_loss = []
        self.total_d_loss = []

    def save(self):
        torch.save(self.generator.state_dict(), 'generator.pth')
        torch.save(self.discriminator.state_dict(), 'discriminator.pth')
    def load(self):
        self.generator.load_state_dict(torch.load('generator.pth'))
        self.discriminator.load_state_dict(torch.load('discriminator.pth'))
    def get_batch(self):
        idx = random.randint(0, 9)

        nums = random.sample(range(1000), 32)
        batch_files = [f"./data/000{idx}/000{n:03d}.png" for n in nums]

        batch = []
        for f in batch_files:
            img = Image.open(f).convert('RGB')
            tensor = self.transform(img)
            batch.append(tensor)
        batch = torch.stack(batch).to(self.device)
        return batch
    def train(self):
        for epoch in range(self.epochs):
            batch = self.get_batch()

            real_target = torch.ones(batch.size(0), 1).to(self.device)
            fake_target = torch.zeros(batch.size(0), 1).to(self.device)

            noise = torch.randn(batch.size(0), self.latent_dim, device=self.device)
            with torch.no_grad():
                gen_img = self.generator(noise)

            d_loss = (self.loss(self.discriminator(gen_img), fake_target) + self.loss(self.discriminator(batch), real_target)) / 2

            gen_img = self.generator(noise)

            val = self.discriminator(gen_img)

            g_loss = self.loss(val, real_target)

            self.optim_g.zero_grad()
            self.optim_d.zero_grad()

            d_loss.backward()
            g_loss.backward()

            self.optim_g.step()
            self.optim_d.step()

            print(f"epoch: {epoch}, g_loss: {g_loss.item():.4f}, d_loss: {d_loss.item():.4f}")
            self.total_g_loss.append(g_loss.item())
            self.total_d_loss.append(d_loss.item())

            self.save()
    def graph(self):
        plt.plot(self.total_g_loss, label="G_Loss")
        plt.plot(self.total_d_loss, label="D_Loss")
        plt.legend()
        plt.show()

gan = GAN(lr=2e-4, latent_dim=100, batch_size=32, epochs=100)
gan.load()
gan.train()
gan.graph()