import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1024 * 512, 128 * 256 * 128),
            nn.ReLU(),
            nn.Unflatten(1,(128, 256, 128)),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh() #Based on transform.Compose
        )
    def forward(self, input):
        img = self.main(input)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential()