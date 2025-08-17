#TODO: generator, discriminator, train loop, chaining blocks, upscaling
from torch import nn
import torch

class PixelNorm(nn.Module):
    def __init__(self, eps = 1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, leak, pix_norm = True, eps=1e-8):
        super(conv_block, self).__init__()
        self.pix_norm = pix_norm

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.pixel_norm = PixelNorm(eps=eps)
        self.leaky_relu = nn.LeakyReLU(leak, True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.pixel_norm(x) if self.pix_norm else x
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.pixel_norm(x) if self.pix_norm else x
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def minibatch_std(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_statistics], dim=1)