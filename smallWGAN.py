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
    def __init__(self, batch_size = 64, latent_dim = 100, use_cuda=True, lr_g = 2e-4, lr_d = 2e-4,g_betas = (0.5, 0.999), d_betas = (0.5, 0.999), c_lambda = 10, transform_size = (128, 64), transform_horizontal_flip_chance = 0.5):
        super(WGAN, self).__init__()

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.c_lambda = c_lambda

        if use_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.generator = Generator(self.latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=g_betas)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=d_betas)

        self.mse = nn.MSELoss()

        self.transform = transforms.Compose([
            transforms.Resize(transform_size),
            transforms.RandomHorizontalFlip(transform_horizontal_flip_chance),
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

    def get_gradient(self, discriminator, real, fake, epsilon):
        mixed_img = real * epsilon + fake * (1 - epsilon)
        vals = discriminator(mixed_img)
        ones = torch.ones_like(vals)

        g = torch.autograd.grad(outputs=vals, inputs=mixed_img, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
        return g
    def get_gradient_penalty(self, gradient):
        g = gradient.view(len(gradient), -1)
        norm = g.norm(2, dim=1)
        return self.mse(norm, torch.ones_like(norm))

    def g_loss(self, d_gen_val):
        g_loss = -torch.mean(d_gen_val)
        return g_loss
    def d_loss(self, d_fake_val, d_real_val, gradient_penalty):
        d_loss = -torch.mean(d_fake_val)
        return d_loss
    def generate(self, x_images = 1, save = False):
        self.generator.eval()
        for i in range(x_images):
            noise = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
            with torch.no_grad():
                img = self.generator(noise)
            if save:
                self.save_img(img, i, rand=True)
        self.generator.train()

    def save_img(self, img, e, rand = False):
        img = img.detach().squeeze().cpu()
        img = (img + 1) / 2
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        if rand:
            i = random.randint(1, 1000)
            Image.fromarray(img).save(f"generatedImages/gen{e}_{i}.png")
        else:
            Image.fromarray(img).save(f"generatedImages/gen{e}.png")