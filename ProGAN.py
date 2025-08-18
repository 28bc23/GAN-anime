#TODO: generator, discriminator, train loop, chaining blocks, upscaling
from torch import nn
import torch

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
        self.bias = nn.Parameter(bias.view(1, self.bias.shape[0], 1,1))
        self.conv.bias = None

        self.scaler = torch.sqrt(gain/(kernel_size**2 * in_channels))

        nn.init.normal_(self.conv.weight)
        nn.init.constant(self.bias, val=0)
    def forward(self, x):
        return self.conv(x * self.scaler) + self.bias

class ELRLinear(nn.Module):
    def __init__(self, in_dim, out_dim, gain = 2):
        super(ELRLinear, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)

        self.bias = self.fc.bias
        self.fc.bias = None

        self.scaler = torch.sqrt(gain/in_dim)

        nn.init.normal_(self.fc.weight)
        nn.init.constant(self.bias, val=0)
    def forward(self, x):
        return self.fc(x * self.scaler) + self.bias

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leak, pix_norm = True, eps=1e-8):
        super(conv_block, self).__init__()
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
        return x.view(x.size[0], self.latent_size, 4, 4)

class block (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(block, self).__init__()
        self.block = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            conv_block(in_channels, out_channels, 4, 1, 3, .2, True),
            conv_block(out_channels, out_channels, 3, 1, 1, .2, True),
        )
    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, latent_size = 512):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.step = 1

        first = nn.Sequential(
            ELRLinear(self.latent_size, self.latent_size*4*4),
            ReshapeLatent(self.latent_size),
            conv_block(self.latent_size, self.latent_size, 4, 1, 3, .2, True),
            conv_block(self.latent_size, self.latent_size, 3, 1, 1, .2, True),
        )

        self.blocks = nn.ModuleList([
            first,                                 #1
            block(512, 512), #2
            block(512, 512), #3
            block(512, 512), #4
            block(512, 256), #5
            block(256, 128), #6
            block(128, 64),  #7
            block(64, 32),   #8
            block(32, 16),   #9
        ])

        self.to_rgb_old = ELRConv(self.latent_size, 3, 3, (1, 2), 1)
        self.to_rgb_new = ELRConv(self.latent_size, 3, 3, (1, 2), 1)
    def extend(self):
        self.step += 1

        if self.step == 5:
            self.to_rgb_old = self.to_rgb_new
            self.to_rgb_new = ELRConv(256, 3, 3, (1, 2), 1)
        elif self.step == 6:
            self.to_rgb_old = self.to_rgb_new
            self.to_rgb_new = ELRConv(128, 3, 3, (1, 2), 1)
        elif self.step == 7:
            self.to_rgb_old = self.to_rgb_new
            self.to_rgb_new = ELRConv(64, 3, 3, (1, 2), 1)
        elif self.step == 8:
            self.to_rgb_old = self.to_rgb_new
            self.to_rgb_new = ELRConv(32, 3, 3, (1, 2), 1)
        elif self.step == 9:
            self.to_rgb_old = self.to_rgb_new
            self.to_rgb_new = ELRConv(16, 3, 3, (1, 2), 1)
    def forward(self, x):
        x_old = None
        for block in range(0, self.step):
            x = self.blocks[block](x)
            if block == self.step-2:
                x_old = x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def minibatch_std(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_statistics], dim=1)