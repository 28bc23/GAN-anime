import os
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import v2 as transforms

from matplotlib import pyplot as plt
from PIL import Image


class PixelNorm(nn.Module):
    def __init__(self, eps = 1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
class ELRConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding, gain = 2):
        super(ELRConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        bias = self.conv.bias
        self.bias = nn.Parameter(bias.view(1, bias.shape[0], 1,1))
        self.conv.bias = None

        self.scaler = (gain / (in_channels * (kernel_size ** 2))) ** 0.5

        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.bias, val=0)
    def forward(self, x):
        return self.conv(x * self.scaler) + self.bias

class ELRLinear(nn.Module):
    def __init__(self, in_dim, out_dim, gain = 2):
        super(ELRLinear, self).__init__()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)

        self.bias = self.fc.bias
        self.fc.bias = None

        self.scaler = (gain/in_dim)**0.5

        nn.init.normal_(self.fc.weight)
        nn.init.constant_(self.bias, val=0)
    def forward(self, x):
        x = self.flat(x)
        return self.fc(x * self.scaler) + self.bias

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leak, pix_norm = True, eps=1e-8):
        super(ConvBlock, self).__init__()
        self.pix_norm = pix_norm

        self.conv = ELRConv(in_channels, out_channels, kernel_size, stride, padding)
        self.pixel_norm = PixelNorm(eps=eps)
        self.leaky_relu = nn.LeakyReLU(leak, True)
    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = self.pixel_norm(x) if self.pix_norm else x
        return x

class ReshapeLatent(nn.Module):
    def __init__(self, latent_size):
        super(ReshapeLatent, self).__init__()
        self.latent_size = latent_size
    def forward(self, x):
        return x.view(x.size(0), self.latent_size, 4, 4)

class BlockG (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockG, self).__init__()
        self.block = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvBlock(in_channels, out_channels, 3, 1, 1, .2, True),
            ConvBlock(out_channels, out_channels, 3, 1, 1, .2, True),
        )
    def forward(self, x):
        x = self.block(x)
        return x

class BlockD (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockD, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, 1, 1, .2, False),
            ConvBlock(in_channels, out_channels, 3, 1, 1, .2, False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        x = self.block(x)
        return x

class MinibatchStd(nn.Module):
    def forward(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_statistics], dim=1)


class Generator(nn.Module):
    def __init__(self, device, latent_size = 512, alpha_addition = 0.01):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.step = 1
        self.alpha = 0
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.alpha_addition = alpha_addition
        self.device = device

        first = nn.Sequential(
            ELRLinear(self.latent_size, self.latent_size * 4 * 4),
            ReshapeLatent(self.latent_size),
            ConvBlock(self.latent_size, self.latent_size, 4, 2, 3, .2, True),
            ConvBlock(self.latent_size, self.latent_size, 3, 1, 1, .2, True)
        )

        self.blocks = nn.ModuleList([
            first,                                 #1
            BlockG(512, 512), #2
            BlockG(512, 512), #3
            BlockG(512, 512), #4
            BlockG(512, 256), #5
            BlockG(256, 128), #6
            BlockG(128, 64),  #7
            BlockG(64, 32),   #8
            BlockG(32, 16),   #9
        ])

        self.to_rgb_old = ELRConv(self.latent_size, 3, 3, (1, 2), 1)
        self.to_rgb_new = ELRConv(self.latent_size, 3, 3, (1, 2), 1)
    def extend(self):
        if self.step != 9:
            self.step += 1

            if self.step == 5:
                self.to_rgb_old = self.to_rgb_new
                self.to_rgb_new = ELRConv(256, 3, 3, (1, 2), 1).to(self.device)
            elif self.step == 6:
                self.to_rgb_old = self.to_rgb_new
                self.to_rgb_new = ELRConv(128, 3, 3, (1, 2), 1).to(self.device)
            elif self.step == 7:
                self.to_rgb_old = self.to_rgb_new
                self.to_rgb_new = ELRConv(64, 3, 3, (1, 2), 1).to(self.device)
            elif self.step == 8:
                self.to_rgb_old = self.to_rgb_new
                self.to_rgb_new = ELRConv(32, 3, 3, (1, 2), 1).to(self.device)
            elif self.step == 9:
                self.to_rgb_old = self.to_rgb_new
                self.to_rgb_new = ELRConv(16, 3, 3, (1, 2), 1).to(self.device)
            self.alpha = 0
    def forward(self, x, alpha):
        self.alpha = alpha
        x_old = None
        for block in range(0, self.step):
            x = self.blocks[block](x)
            if block == self.step-2:
                x_old = self.upsample(x)
        if x_old is not None:
            x_old = self.to_rgb_old(x_old)
            x = self.to_rgb_new(x)
            out = x_old * (1 - self.alpha) + x * self.alpha
        else:
            out = self.to_rgb_new(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, device, alpha_addition = 0.01):
        super(Discriminator, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.step = 1
        self.alpha = 0
        self.alpha_addition = alpha_addition
        self.device = device

        self.from_rgb = nn.ModuleList([]).to(self.device)

        self.resolution = (4, 2)
        self.transform = transforms.Compose([
            transforms.Resize((4, 2)),
            transforms.RandomHorizontalFlip(.5),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        out_channels = 512
        y_size = 2
        for i in range(0, 9):
            if i <= 3:
                new_conv = ELRConv(3, out_channels, 3, 1, (1, int(y_size/2 + 1)))
                y_size = y_size * 2
            else:
                out_channels = out_channels / 2
                new_conv = ELRConv(3, int(out_channels), 3, 1, (1, int(y_size/2 + 1)))
                y_size = y_size * 2
            self.from_rgb.append(new_conv)

        self.rectTosqar = ELRConv(3, 3, 3, 1, (1, int(512/2 + 1)))

        first = nn.Sequential(
            ConvBlock(3, 16, 1, 1, 0, .2, False),
            ConvBlock(16, 16, 3, 1, 1, .2, False),
            ConvBlock(16, 32, 3, 1, 1, .2, False),
            self.downsample
        ).to(self.device)
        last = nn.Sequential(
            MinibatchStd(),
            ConvBlock(513, 512, 3, 1, 1, .2, False),
            ConvBlock(512, 512, 4, 1, 0, .2, False),
            nn.Flatten(),
            ELRLinear(512, 1)
        ).to(self.device)
        self.chain = nn.ModuleList([
            last,
        ]).to(self.device)

        self.blocks = nn.ModuleList([
            BlockD(512, 512),  # 2
            BlockD(512, 512),  # 3
            BlockD(512, 512),  # 4
            BlockD(256, 512),  # 5
            BlockD(128, 256),  # 6
            BlockD(64, 128),   # 7
            BlockD(32, 64),    # 8
            first                                    # 9
        ]).to(self.device)
    def extend(self):
        if self.step != 9:
            self.step += 1
            self.chain.insert(0, self.blocks.pop(0))
            self.alpha = 0
            self.resolution = (self.resolution[0]*2, self.resolution[1]*2)
            self.transform = transforms.Compose([
                transforms.Resize(self.resolution),
                transforms.RandomHorizontalFlip(.5),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return self.transform
    def forward(self, x, alpha):
        self.alpha = alpha
        if self.step != 9:
            if self.step != 1:
                x_old = self.downsample(x)
                x_old = self.from_rgb[self.step-2](x_old)

                x = self.from_rgb[self.step-1](x)
            else:
                x = self.from_rgb[self.step-1](x)
                x_old = self.downsample(x)
        else:
            x_old = self.downsample(x)
            x_old = self.from_rgb[self.step - 2](x_old)
            x = self.rectTosqar(x)

        for i, block in enumerate(self.chain):
            x = block(x)
            if i == 0 and self.step != 1:
                x = x_old * (1 - self.alpha) + x * self.alpha

        out = x
        return out
    def get_transform(self):
        return self.transform



class ProGAN:
    def __init__(self,load = True, epochs = 5, save_step = 10, g_itter = 1, d_itter = 2, batch_size = 128, latent_size = 512, alpha_addition = 0, cuda = True, lambda_gp = 10, lr_g = 2e-4, lr_d = 2e-4, g_betas = (0.5, 0.999), d_betas = (0.5, 0.999)):

        self.batch_size = batch_size
        self.epochs = epochs
        self.start_epochs = 0
        self.latent_size = latent_size
        self.lambda_gp = lambda_gp
        self.d_itter = d_itter
        self.g_itter = g_itter
        self._load = load
        self.save_step = save_step

        self.alpha_addition = alpha_addition
        self.d_alpha_addition = alpha_addition / self.d_itter
        self.g_alpha_addition = alpha_addition / self.g_itter

        if cuda:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        print("device: ", self.device)

        self.generator = Generator(latent_size = self.latent_size, alpha_addition=self.g_alpha_addition, device = self.device).to(self.device)
        self.discriminator = Discriminator(alpha_addition=self.d_alpha_addition, device = self.device).to(self.device)

        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=g_betas)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=d_betas)

        self.dataset = self.get_dataset()


        ### graph data ###
        self.epoch_losses_g = []
        self.epoch_losses_d = []

        self.step_losses_g = []
        self.step_losses_d = []

        self.itter_losses_g = []
        self.itter_losses_d = []

        self.d_real_values = []
        self.d_fake_values = []

        self.g_values = []
    def get_dataset(self):
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        dataset = datasets.ImageFolder(root="./data", transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        return dataloader

    def save(self, epoch):
        epoch = epoch + self.start_epochs
        path = f"./pre-trainedModels/large/ProGAN/{epoch}"
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.generator.state_dict(), f"{path}/generator_epoch_{epoch}.pth")
        torch.save(self.discriminator.state_dict(), f"{path}/discriminator_epoch_{epoch}.pth")
        print("--- saved ---")
    def load(self, epoch = None):
        if epoch is None:
            path = f"./pre-trainedModels/large/ProGAN"
            bigger_num = 0
            for path in os.scandir(path):
                number = path.name
                if number.isdigit():
                    number = int(number)
                    if number > bigger_num:
                        bigger_num = number
            if bigger_num != 0:
                path = f"{path}/{bigger_num}"
                self.generator.load_state_dict(torch.load(f"{path}/generator_epoch_{bigger_num}.pth"))
                self.discriminator.load_state_dict(torch.load(f"{path}/discriminator_epoch_{bigger_num}.pth"))
                self.start_epochs = bigger_num
                print("--- loaded ---")
    def graph(self):
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        fig.suptitle("Training metrics", fontsize=16)

        # Epoch losses
        axes[0].plot(self.epoch_losses_g, label="Generator epoch mean Loss")
        axes[0].plot(self.epoch_losses_d, label="Discriminator epoch mean Loss")
        axes[0].set_title("Epoch losses")
        axes[0].legend()
        axes[0].grid(True)

        # Step losses
        axes[1].plot(self.step_losses_g, label="Generator step mean Loss")
        axes[1].plot(self.step_losses_d, label="Discriminator step mean Loss")
        axes[1].set_title("Step losses")
        axes[1].legend()
        axes[1].grid(True)

        # Itter losses
        axes[2].plot(self.itter_losses_g, label="Generator itter mean Loss")
        axes[2].plot(self.itter_losses_d, label="Discriminator itter mean Loss")
        axes[2].set_title("Itter losses")
        axes[2].legend()
        axes[2].grid(True)

        # Values
        axes[3].plot(self.d_real_values, label="Discriminator Real values")
        axes[3].plot(self.d_fake_values, label="Discriminator Fake values")
        axes[3].plot(self.g_values, label="Generator values")
        axes[3].set_title("Values")
        axes[3].legend()
        axes[3].grid(True)

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        plt.show()

    def compute_gradient_penalty(self, real_samples, fake_samples, d_alpha):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_samples.size(0)

        # Create interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

        # Get discriminator output
        d_interpolates = self.discriminator(interpolates, d_alpha)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Flatten gradients
        gradients = gradients.view(batch_size, -1)

        # Compute gradient penalty: (||gradients||_2 - 1)^2
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self):
        if self._load:
            self.load()
        transform = self.discriminator.get_transform()
        d_alpha = 0
        g_alpha = 0
        extend_level = 0

        self.generator.train()
        self.discriminator.train()

        if self.alpha_addition == 0:
            self.alpha_addition = 1 / len(self.dataset)
            print(f"--- Alpha addition was set to 0 using calculated value: {self.alpha_addition} ---")

        for epoch in range(self.epochs):
            step_losses_g = []
            step_losses_d = []
            for step, (batch, label) in enumerate(self.dataset):

                if d_alpha + g_alpha >= 2 and extend_level != 10:
                    transform = self.discriminator.extend()
                    self.generator.extend()
                    d_alpha = 0
                    g_alpha = 0
                    extend_level += 1
                    print("--- extended ---")

                batch = transform(batch)
                batch = batch.to(self.device)


                ### Discriminator optim ###
                ext, ext2 = False, False
                itter_d_loss = []
                for i in range(self.d_itter):
                    self.optim_d.zero_grad()

                    noise = torch.randn(batch.size(0), self.latent_size, 1, 1).to(self.device)
                    fake = self.generator(noise, g_alpha)
                    fake = fake.detach()

                    real_val = self.discriminator(batch, d_alpha)
                    fake_val = self.discriminator(fake, d_alpha)

                    gp = self.compute_gradient_penalty(batch, fake, d_alpha).to(self.device)

                    d_loss = torch.mean(fake_val) - torch.mean(real_val) + self.lambda_gp * gp
                    d_loss.backward()
                    self.optim_d.step()
                    
                    log_real_val = torch.mean(real_val.detach()).item()
                    log_fake_val = torch.mean(fake_val.detach()).item()
                    log_d_loss = d_loss.item()
                    
                    print(f"Discriminator, epoch: {epoch}, step: {step}, itter: {i}, batch: {step}/{len(self.dataset)}, d_loss: {log_d_loss}, gp: {gp}, real_val: {log_real_val}, fake_val: {log_fake_val}, extend level: {extend_level}, alpha: {d_alpha}")
                    
                    self.d_real_values.append(log_real_val)
                    self.d_fake_values.append(log_fake_val)
                    self.itter_losses_d.append(log_d_loss)
                    itter_d_loss.append(log_d_loss)
                    step_losses_d.append(log_d_loss)
                self.step_losses_d.append(np.mean(itter_d_loss).item())
                d_alpha += self.alpha_addition

                ### Generator optim ###
                ext = False
                itter_g_loss = []
                for i in range(self.g_itter):
                    self.optim_g.zero_grad()
                    noise = torch.randn(batch.size(0), self.latent_size, 1, 1).to(self.device)
                    fake = self.generator(noise, g_alpha)
                    fake_val = self.discriminator(fake, d_alpha)
                    g_loss = -torch.mean(fake_val)
                    g_loss.backward()
                    self.optim_g.step()
                    
                    log_g_loss = g_loss.item()
                    log_fake_val = torch.mean(fake_val.detach()).item()
                    
                    print(f"Generator, epoch: {epoch}, step: {step}, itter: {i}, batch: {step}/{len(self.dataset)}, g_loss: {log_g_loss}, val: {log_fake_val}, extend level: {extend_level}, alpha: {g_alpha}")
                    
                    self.g_values.append(log_fake_val)
                    self.itter_losses_g.append(log_g_loss)
                    itter_g_loss.append(log_g_loss)
                    step_losses_g.append(log_g_loss)
                self.step_losses_g.append(np.mean(itter_g_loss).item())
                g_alpha += self.alpha_addition

                if step % self.save_step == 0:
                    self.save(epoch)
            self.epoch_losses_g.append(np.mean(step_losses_g).item())
            self.epoch_losses_d.append(np.mean(step_losses_d).item())
            self.save(self.epochs)
            self.graph()


progan = ProGAN(alpha_addition=1, cuda=False, batch_size=1, epochs=1)
progan.train()