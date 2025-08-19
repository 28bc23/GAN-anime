#TODO: generator, discriminator, train loop, chaining blocks, upscaling
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets, transforms
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
        return x.view(x.size(0), self.latent_size, 4, 4)

class blockG (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(blockG, self).__init__()
        self.block = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            conv_block(in_channels, out_channels, 3, 1, 1, .2, True),
            conv_block(out_channels, out_channels, 3, 1, 1, .2, True),
        )
    def forward(self, x):
        x = self.block(x)
        return x

class blockD (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(blockD, self).__init__()
        self.block = nn.Sequential(
            conv_block(in_channels, in_channels, 3, 1, 1, .2, False),
            conv_block(in_channels, out_channels, 3, 1, 1, .2, False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        x = self.block(x)
        return x

class minibatch_std(nn.Module):
    def forward(self, x):
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_statistics], dim=1)


class Generator(nn.Module):
    def __init__(self, latent_size = 512, alpha_addition = 0.01):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.step = 1
        self.alpha = 0
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.alpha_addition = alpha_addition

        first = nn.Sequential(
            ELRLinear(self.latent_size, self.latent_size * 4 * 4),
            ReshapeLatent(self.latent_size),
            conv_block(self.latent_size, self.latent_size, 4, 2, 3, .2, True),
            conv_block(self.latent_size, self.latent_size, 3, 1, 1, .2, True)
        )

        self.blocks = nn.ModuleList([
            first,                                 #1
            blockG(3, 512), #2
            blockG(512, 512), #3
            blockG(512, 512), #4
            blockG(512, 256), #5
            blockG(256, 128), #6
            blockG(128, 64),  #7
            blockG(64, 32),   #8
            blockG(32, 16),   #9
        ])

        self.to_rgb_old = ELRConv(self.latent_size, 3, 3, (1, 2), 1)
        self.to_rgb_new = ELRConv(self.latent_size, 3, 3, (1, 2), 1)
    def extend(self):
        if self.step != 9:
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
            self.alpha = 0
    def forward(self, x):
        if self.alpha == 1:
            self.extend()
        x_old = None
        for block in range(0, self.step):
            x = self.blocks[block](x)
            if block == self.step-2:
                x_old = self.upsample(x)
        if x_old is not None:
            x_old = self.to_rgb_old(x_old)
            x = self.to_rgb_new(x)
            print("alpha: ", self.alpha)
            print("x_old: ", x_old.shape)
            print("x: ", x.shape)
            out = x_old * (1 - self.alpha) + x * self.alpha
            self.alpha += self.alpha_addition
        else:
            print("x")
            out = self.to_rgb_new(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, alpha_addition = 0.01):
        super(Discriminator, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.step = 1
        self.alpha = 0
        self.alpha_addition = alpha_addition

        self.from_rgb = nn.ModuleList([])

        self.resolution = (4, 2)
        self.transform = transforms.Compose([
            transforms.Resize((4, 2)),
            transforms.RandomHorizontalFlip(.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        out_channels = 512
        y_size = 2
        for i in range(0, 9):
            if i <= 3:
                new_conv = ELRConv(3, out_channels, 3, 1, (1, y_size/2 + 1))
                y_size = y_size * 2
            else:
                out_channels = out_channels / 2
                new_conv = ELRConv(3, out_channels, 3, 1, (1, y_size/2 + 1))
                y_size = y_size * 2
            self.from_rgb.append(new_conv)

        first = nn.Sequential(
            conv_block(3, 16, 1, 1, 0, .2, False),
            conv_block(16, 16, 3, 1, 1, .2, False),
            conv_block(16, 32, 3, 1, 1, .2, False),
            self.downsample()
        )
        last = nn.Sequential(
            minibatch_std(),
            conv_block(513, 512, 3, 1, 1, .2, False),
            conv_block(512, 512, 4, 1, 0, .2, False),
            nn.Flatten(),
            ELRLinear(512, 1)
        )
        self.chain = nn.ModuleList([
            last()
        ])

        self.blocks = nn.ModuleList([
            blockD(512, 512),  # 2
            blockD(512, 512),  # 3
            blockD(512, 512),  # 4
            blockD(256, 512),  # 5
            blockD(128, 256),  # 6
            blockD(64, 128),   # 7
            blockD(32, 64),    # 8
            first()                                  # 9
        ])
    def extend(self):
        if self.step != 9:
            self.step += 1
            self.chain.insert(0, self.blocks.pop(0))
            self.alpha = 0
            self.transform = transforms.Compose([
                transforms.Resize(self.resolution),
                transforms.RandomHorizontalFlip(.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.resolution = self.resolution * 2
    def forward(self, x):
        if self.alpha == 1:
            self.extend()

        x = self.from_rgb[self.step-1](x)
        x_old = self.downsample(x)
        for block in self.chain:
            x = block(x)

        if self.step != 1:
            out = x_old * (1 - self.alpha) + x * self.alpha
            self.alpha += self.alpha_addition
        else:
            out = x

        return out, self.transform
    def get_transform(self):
        return self.transform



class ProGAN:
    def __init__(self, epochs = 5, batch_size = 128):

        self.batch_size = batch_size
        self.epochs = epochs

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.dataset = self.get_dataset()
    def get_dataset(self):
        dataset = datasets.ImageFolder(root="./data")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        return dataloader
    def train(self):
        transform = self.discriminator.get_transform()
        for epoch in range(self.epochs):
            for i, batch in enumerate(self.dataset):
                batch = transform(batch)