import torch
from torch import nn
import random
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

manualSeed = 24150
random.seed(manualSeed)
torch.manual_seed(manualSeed)

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
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, leak):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(leak, True)
        )
    def forward(self, x):
        return self.block(x)

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()


def save_img(img, e, rand=False):
    img = img.detach().squeeze().cpu()
    img = (img + 1) / 2
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    if rand:
        i = random.randint(1, 1000)
        Image.fromarray(img).save(f"generatedImages/WGAN/gen{e}_{i}.png")
    else:
        Image.fromarray(img).save(f"generatedImages/WGAN/gen{e}.png")

def gen_progress(generator, fixed_noise, epoch):
    generator.eval()
    with torch.no_grad():
        fake_img = generator(fixed_noise).detach().cpu()
    save_img(fake_img, epoch)
    generator.train()

def graph(g_loss, d_loss, gp = None):
    plt.plot(g_loss, label="G_Loss")
    plt.plot(d_loss, label="D_Loss")
    plt.plot(gp, label="gradient penalty")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def get_batch(batch_size, transform, device):
    idx = random.randint(0, 39)

    nums = random.sample(range(1000), batch_size)
    batch_files = [f"./data/00{idx:02d}/000{n:03d}.png" for n in nums]

    batch = []
    for f in batch_files:
        img = Image.open(f).convert('RGB')
        tensor = transform(img)
        batch.append(tensor)
    batch = torch.stack(batch).to(device)
    return batch

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)