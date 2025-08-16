import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

from utils.utilsGAN import ConvTransBlock, ConvBlock, gen_progress, save_img, graph, get_batch

manualSeed = 24150
random.seed(manualSeed)
torch.manual_seed(manualSeed)




class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ConvTransBlock(in_channel=latent_dim, out_channel=512, kernel_size=4, stride=1, padding=0, output_padding=0),
            ConvTransBlock(in_channel=512, out_channel=256, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTransBlock(in_channel=256, out_channel=128, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTransBlock(in_channel=128, out_channel=64, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTransBlock(in_channel=64, out_channel=32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ConvTranspose2d(32, 3, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            ConvBlock(in_channel=3, out_channel=32, kernel_size=4, stride=2, padding=1, leak=0.2), # 128x64; 64x32
            ConvBlock(in_channel=32, out_channel=64, kernel_size=4, stride=2, padding=1, leak=0.2), # 64x32; 32x16
            ConvBlock(in_channel=64, out_channel=128, kernel_size=4, stride=2, padding=1, leak=0.2), # 32x16; 16x8
            ConvBlock(in_channel=128, out_channel=256, kernel_size=4, stride=2, padding=1, leak=0.2), # 16x8; 8x4
            ConvBlock(in_channel=256, out_channel=512, kernel_size=4, stride=2, padding=1, leak=0.2), # 8x4; 4x2
            nn.Conv2d(512, 1, kernel_size=(4, 2), stride=1, padding=0),  # 4x2; 1x1
        )
    def forward(self, x):
        return self.main(x)

class WGAN:
    def __init__(self, epochs = 1000, batch_size = 64, latent_dim = 100, save_freq = 10, test_gen_freq = 10, use_cuda=True, g_steps = 1, d_steps = 3,lr_g = 2e-4, lr_d = 2e-4,g_betas = (0.5, 0.999), d_betas = (0.5, 0.999), c_lambda = 10, transform_size = (128, 64), transform_horizontal_flip_chance = 0.5):
        super(WGAN, self).__init__()

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.c_lambda = c_lambda
        self.epochs = epochs
        self.save_freq = save_freq
        self.test_gen_freq = test_gen_freq

        if use_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.generator = Generator(self.latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=g_betas)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=d_betas)

        self.g_steps = g_steps
        self.d_steps = d_steps

        self.mse = nn.MSELoss()

        self.fixed_noise = torch.randn(1, self.latent_dim, 1, 1, device=self.device)

        self.transform = transforms.Compose([
            transforms.Resize(transform_size),
            transforms.RandomHorizontalFlip(transform_horizontal_flip_chance),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.d_loss_data = []
        self.g_loss_data = []

    def save(self):
        torch.save(self.generator.state_dict(), './gWGAN.pth')
        torch.save(self.discriminator.state_dict(), './dWGAN.pth')
        print("--- saved ---")

    def load(self):
        self.generator.load_state_dict(torch.load('./gWGAN.pth'))
        self.discriminator.load_state_dict(torch.load('./dWGAN.pth'))
        print("--- loaded ---")

    def get_gradient(self, discriminator, real, fake):
        epsilon = torch.rand(self.batch_size, 1, 1, 1, device=self.device, requires_grad=True)
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
        d_loss = torch.mean(d_fake_val) - torch.mean(d_real_val) + self.c_lambda * gradient_penalty
        return d_loss
    def generate(self, x_images = 1, save = False):
        self.generator.eval()
        for i in range(x_images):
            noise = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
            with torch.no_grad():
                img = self.generator(noise)
            if save:
                save_img(img, i, rand=True)
        self.generator.train()

    def train(self):
        for e in range(self.epochs):
            batch = get_batch(batch_size=self.batch_size, transform=self.transform, device=self.device).to(self.device)

            #############################
            ##   Discriminator optim   ##
            #############################
            mean_d_loss = 0
            for _ in range(self.d_steps):
                self.optim_d.zero_grad()

                noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
                fake = self.generator(noise).detach().to(self.device)

                real_val = self.discriminator(batch).to(self.device)
                fake_val = self.discriminator(fake).to(self.device)

                gradient = self.get_gradient(discriminator=self.discriminator, real=batch, fake=fake)
                gradient_penalty = self.get_gradient_penalty(gradient)
                d_loss = self.d_loss(d_fake_val=fake_val, d_real_val=real_val, gradient_penalty=gradient_penalty)

                d_loss.backward()
                self.optim_d.step()
                mean_d_loss += d_loss.item()


            #############################
            ##     Generator optim     ##
            #############################
            mean_g_loss = 0
            for _ in range(self.g_steps):
                self.optim_g.zero_grad()

                noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
                fake_img = self.generator(noise)
                fake_val = self.discriminator(fake_img).to(self.device)

                g_loss = self.g_loss(fake_val)

                g_loss.backward()
                self.optim_g.step()
                mean_g_loss += g_loss.item()

            ### Data collection ###
            mean_g_loss = mean_g_loss / self.g_steps
            mean_d_loss = mean_d_loss / self.d_steps
            self.d_loss_data.append(mean_d_loss)
            self.g_loss_data.append(mean_g_loss)

            if e % self.save_freq == 0:
                self.save()
            if e % self.test_gen_freq == 0:
                gen_progress(generator=self.generator, fixed_noise=self.fixed_noise, epoch=e)

            print(f"epoch: {e}/{self.epochs}, "
                  f"g_loss: {mean_g_loss:.4f}, "
                  f"d_loss: {mean_d_loss:.4f}")
        self.save()
        gen_progress(generator=self.generator, fixed_noise=self.fixed_noise, epoch=self.epochs)
        graph(g_loss=self.g_loss_data, d_loss=self.d_loss_data)


### RUN ###
wgan = WGAN(epochs=10000, batch_size=128)
print(wgan.device)

wgan.load()
wgan.train()
wgan.generate()