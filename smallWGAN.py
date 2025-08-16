import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

from utils.utilsGAN import ConvTransBlock, ConvBlock, gen_progress, save_img, graph, get_batch, weights_init

manualSeed = 24150
random.seed(manualSeed)
torch.manual_seed(manualSeed)




class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ConvTransBlock(in_channel=latent_dim, out_channel=1024, kernel_size=4, stride=1, padding=0, output_padding=0),
            ConvTransBlock(in_channel=1024, out_channel=512, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTransBlock(in_channel=512, out_channel=256, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTransBlock(in_channel=256, out_channel=128, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTransBlock(in_channel=128, out_channel=64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ConvTranspose2d(64, 3, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
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
    def __init__(self, epochs = 1000, batch_size = 64, latent_dim = 100, save_freq = 10, test_gen_freq = 10, use_cuda=True, g_steps = 1, d_steps = 3,lr_g = 1e-4, lr_d = 1e-4,g_betas = (0.5, 0.999), d_betas = (0.5, 0.999), grad_max_norm = 1.0, c_lambda = 10, transform_size = (128, 64), transform_horizontal_flip_chance = 0.5):
        super(WGAN, self).__init__()

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.c_lambda = c_lambda
        self.epochs = epochs
        self.save_freq = save_freq
        self.test_gen_freq = test_gen_freq
        self.grad_max_norm = grad_max_norm

        if use_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.generator = Generator(self.latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=g_betas)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=d_betas)

        self.g_steps = g_steps
        self.d_steps = d_steps

        self.fixed_noise = torch.randn(1, self.latent_dim, 1, 1, device=self.device)

        self.transform = transforms.Compose([
            transforms.Resize(transform_size),
            transforms.RandomHorizontalFlip(transform_horizontal_flip_chance),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.d_loss_data = []
        self.g_loss_data = []
        self.gp_data = []

    def save(self):
        torch.save(self.generator.state_dict(), './gWGAN.pth')
        torch.save(self.discriminator.state_dict(), './dWGAN.pth')
        print("--- saved ---")

    def load(self):
        self.generator.load_state_dict(torch.load('./gWGAN.pth'))
        self.discriminator.load_state_dict(torch.load('./dWGAN.pth'))
        print("--- loaded ---")

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP"""
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        
        fake = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

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
            gp = 0
            for i in range(self.d_steps):
                self.optim_d.zero_grad()

                noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
                fake = self.generator(noise).detach()

                real_val = self.discriminator(batch)
                fake_val = self.discriminator(fake)

                gradient_penalty = self.compute_gradient_penalty(batch, fake)
                d_loss = self.d_loss(fake_val, real_val, gradient_penalty)

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.grad_max_norm)
                self.optim_d.step()
                
                mean_d_loss += d_loss.item()
                gp += gradient_penalty.item()
                print(f"epoch: {e}/{self.epochs}, step: {i+1}/{self.d_steps}, d_loss: {d_loss.item():.4f}, "
                      f"gp: {gp:.4f}, fake_val: {torch.mean(fake_val.detach()):.4f}, real_val: {torch.mean(real_val.detach()):.4f}")


            #############################
            ##     Generator optim     ##
            #############################
            mean_g_loss = 0
            for i in range(self.g_steps):
                self.optim_g.zero_grad()

                noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
                fake_img = self.generator(noise)
                fake_val = self.discriminator(fake_img)

                g_loss = self.g_loss(fake_val)

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.grad_max_norm)
                self.optim_g.step()
                mean_g_loss += g_loss.item()
                print(f"epoch: {e}/{self.epochs}, step: {i+1}/{self.g_steps}, g_loss: {g_loss.item():.4f}")

            ### Data collection ###
            mean_g_loss = mean_g_loss / self.g_steps
            mean_d_loss = mean_d_loss / self.d_steps
            mean_gp = gp / self.d_steps
            self.d_loss_data.append(mean_d_loss)
            self.g_loss_data.append(mean_g_loss)
            self.gp_data.append(mean_gp)

            if e % self.save_freq == 0:
                self.save()
            if e % self.test_gen_freq == 0:
                gen_progress(generator=self.generator, fixed_noise=self.fixed_noise, epoch=e)
        self.save()
        gen_progress(generator=self.generator, fixed_noise=self.fixed_noise, epoch=self.epochs)
        graph(g_loss=self.g_loss_data, d_loss=self.d_loss_data, gp=self.gp_data)


### RUN ###
wgan = WGAN(epochs=5000, batch_size=128, g_steps=1, d_steps=3, lr_d=1e-4, lr_g=1e-4, grad_max_norm=1.0)
print(wgan.device)

#wgan.load()
wgan.train()
wgan.generate()