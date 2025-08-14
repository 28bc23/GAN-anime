import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from PIL import Image
import torch.nn.functional as F

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 2),
            nn.ReLU(),
            nn.Unflatten(1,(512, 4, 2)),
        )

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.block4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.block5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.block6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, input):
        x = self.fc(input)

        x = self.block(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        img = self.final(x)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 8, 3, 2,padding=1)), #1024 -> 512; 512 -> 256
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(8, 16, 3, 2,padding=1)), #512 -> 256; 256 -> 128
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(16, 32, 3, 2, padding=1)), #256 -> 128; 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, 2,padding=1)), #128 -> 64; 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 2, padding=1)), #64 -> 32; 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 2, padding=1)), #32 -> 16; 16 -> 8
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, 2, padding=1)), #16 -> 8; 8 -> 4
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(512, 1024, 3, 2, padding=1)), #8 -> 4; 4 -> 2
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(1024, 1024, 3, 1, padding=1)),  # 4 -> 4; 2 -> 2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(1024 * 4 * 2, 1),
        )
    def forward(self, input):
        val = self.main(input)
        return val

class GAN:
    def __init__(self, lr_g, lr_d, latent_dim, batch_size, epochs):
        super(GAN, self).__init__()

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = Generator(self.latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.weights_init(self.generator)
        self.weights_init(self.discriminator)

        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.0, 0.99))
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.0, 0.99))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.total_g_loss = []
        self.total_d_loss = []

        self.real_val = []
        self.fake_val = []

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def save(self):
        torch.save(self.generator.state_dict(), 'generator.pth')
        torch.save(self.discriminator.state_dict(), 'discriminator.pth')
        print("---- saved ----")
    def load(self):
        self.generator.load_state_dict(torch.load('generator.pth'))
        self.discriminator.load_state_dict(torch.load('discriminator.pth'))
        print("---- loaded ----")
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
    def train(self):
        for epoch in range(self.epochs):
            batch = self.get_batch()
            # ---- Discriminator optim ----

            self.optim_d.zero_grad()

            noise_scale = 0.01
            real_in = (batch + noise_scale * torch.randn_like(batch)).clone().detach().requires_grad_(True)

            noise = torch.randn(batch.size(0), self.latent_dim, device=self.device)
            gen_img = self.generator(noise).detach()
            gen_in = gen_img + noise_scale * torch.randn_like(gen_img)

            out_real = self.discriminator(real_in)
            out_fake = self.discriminator(gen_in)

            d_loss = torch.mean(F.relu(1.0 - out_real)) + torch.mean(F.relu(1.0 + out_fake))
            grad_real = torch.autograd.grad(outputs=out_real.sum(), inputs=real_in, create_graph=True)[0]
            r1 = (grad_real.view(grad_real.size(0), -1).pow(2).sum(1)).mean()
            d_loss = d_loss + 0.5 * 10 * r1

            d_loss.backward()
            self.optim_d.step()

            self.fake_val.append(out_fake.detach().cpu().mean().item())
            self.real_val.append(out_real.detach().cpu().mean().item())

            # ---- Generator optim ----
            self.optim_g.zero_grad()
            noise = torch.randn(batch.size(0), self.latent_dim, device=self.device)
            fake_images = self.generator(noise)
            g_loss = -torch.mean(self.discriminator(fake_images))
            g_loss.backward()
            self.optim_g.step()

            # ---- Graph data ----
            print(f"epoch: {epoch}, g_loss: {g_loss.item():.4f}, d_loss: {d_loss.item():.4f}"
                  f", out_real mean/std: {out_real.detach().cpu().mean().item()}/"
                  f"{out_real.detach().cpu().std().item()}"
                  f", out_fake mean/std: {out_fake.detach().cpu().mean().item()}/"
                  f"{out_fake.detach().cpu().std().item()}")

            self.total_g_loss.append(g_loss.item())
            self.total_d_loss.append(d_loss.item())

            if epoch % 10 == 0:
                self.save()

            if epoch % 5 == 0:
                self.save_img(fake_images[0], epoch)
        self.graph()
    def graph(self):
        plt.plot(self.total_g_loss, label="G_Loss")
        plt.plot(self.total_d_loss, label="D_Loss")
        plt.plot(self.real_val, label="Real_Label")
        plt.plot(self.fake_val, label="Fake_Label")
        plt.legend()
        plt.show()
    def generate(self):
        with torch.no_grad():
            noise = torch.randn(1, self.latent_dim, device=self.device)
            img = self.generator(noise)
        return img
    def save_img(self, img, e):
        img = img.detach()
        img = (img + 1) / 2
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(f"generatedImages/gen{e}.png")

gan = GAN(lr_g=2e-4,lr_d=1e-4, latent_dim=100, batch_size=8, epochs=10000)
print(gan.device)

load = input("Wanna load model?[Y/n]: ")
if load.lower() != "n":
    gan.load()

train = input("Wanna train model?[Y/n]: ")
if train.lower() != "n":
    gan.train()

gen = input("Wanna generate sample?[Y/n]: ")
if gen.lower() != "n":
    img = gan.generate()
    img = (img + 1) / 2
    plt.imshow(img.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()