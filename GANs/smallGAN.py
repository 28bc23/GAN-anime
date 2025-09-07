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
#torch.use_deterministic_algorithms(True)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # latent_dim x 1 x 1 -> 512 x 4 x 4
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 512 x 4 x 4 -> 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 256 x 8 x 8 -> 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64 x 32 x 32 -> 32 x 64 x 64
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 4 x 512 x 512 -> 3 x 1024 x 512
            nn.ConvTranspose2d(64, 3, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        img = self.main(input)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2,padding=1), #128 -> 64; 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, padding=1), #64 -> 32; 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, padding=1), #32 -> 16; 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, padding=1), #16 -> 8; 8 -> 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, padding=1), #8 -> 4; 4 -> 2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=(4,2), stride=1, padding=0),  # 4 -> 1; 2 -> 1
            nn.Sigmoid()
        )
    def forward(self, input):
        val = self.main(input)
        return val

class GAN:
    def __init__(self, lr_g, lr_d, latent_dim, batch_size, steps):
        super(GAN, self).__init__()

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.steps = steps

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = Generator(self.latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.optim_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        self.loss = nn.BCELoss()

        self.real_label = 0.9
        self.fake_label = 0.1

        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.RandomHorizontalFlip(.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.d_steps = 20

        self.fixed_noise = torch.randn(1, self.latent_dim, 1, 1, device=self.device)

        self.total_g_loss = []
        self.total_d_loss = []

    def models_init(self):
        self.weights_init(self.generator)
        self.weights_init(self.discriminator)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        print(f"init {classname}")

    def save(self):
        torch.save(self.generator.state_dict(), 'generator.pth')
        torch.save(self.discriminator.state_dict(), 'discriminator.pth')
        print("---- saved ----")
    def load(self):
        self.generator.load_state_dict(torch.load('generator.pth'))
        self.discriminator.load_state_dict(torch.load('discriminator.pth'))
        print("---- loaded ----")
    def get_batch(self):
        idx = random.randint(0, 39)

        nums = random.sample(range(1000), self.batch_size)
        batch_files = [f"./data/00{idx:02d}/000{n:03d}.png" for n in nums]

        batch = []
        for f in batch_files:
            img = Image.open(f).convert('RGB')
            tensor = self.transform(img)
            batch.append(tensor)
        batch = torch.stack(batch).to(self.device)
        return batch
    def train(self):
        for step in range(self.steps):
            batch = self.get_batch()
            r_d_m = 0
            f_d_m = 0
            d_loss_m = 0
            for _ in range(self.d_steps):
                # ---- Discriminator optim ----

                # -- real --
                self.optim_d.zero_grad()
                label = torch.full((batch.shape[0],), self.real_label,dtype=torch.float, device=self.device)
                #batch += torch.randn_like(batch) * 0.05
                r_out = self.discriminator(batch).view(-1)
                d_loss_r = self.loss(r_out, label)
                d_loss_r.backward()
                r_d = r_out.mean().item()
                r_d_m += r_d
                # -- fake --
                noise = torch.randn(batch.size(0), self.latent_dim, 1, 1, device=self.device)
                f = self.generator(noise)
                #fake_img = f.detach() + torch.randn_like(f) * 0.05
                label.fill_(self.fake_label)
                f_out = self.discriminator(f.detach()).view(-1)
                d_loss_f = self.loss(f_out, label)
                d_loss_f.backward()
                f_d = f_out.mean().item()
                f_d_m += f_d
                d_loss = d_loss_r + d_loss_f
                self.optim_d.step()
                d_loss_m += d_loss.item()

            fg_d_m = 0
            g_loss_m = 0
            i = 0
            while True:
                # ---- Generator optim ----
                self.optim_g.zero_grad()
                label.fill_(self.real_label)
                noise = torch.randn(batch.size(0), self.latent_dim, 1, 1, device=self.device)
                f = self.generator(noise)
                out = self.discriminator(f).view(-1)
                g_loss = self.loss(out, label)
                g_loss.backward()
                fg_d = out.mean().item()
                fg_d_m += fg_d
                i += 1
                self.optim_g.step()
                g_loss_m += g_loss.item()
                if g_loss.item() < 0.45:
                    break

            # ---- Graph data ----
            print(f"step: {step}/{self.steps}, "
                  f"g_loss: {g_loss_m/i:.4f}, "
                  f"d_loss: {d_loss_m/self.d_steps:.4f}, "
                  f"d_real_detect: {r_d_m/self.d_steps}, "
                  f"d_fake_detect: {f_d/self.d_steps}, "
                  f"d_fake_detect_v2: {fg_d_m/i}")

            self.total_g_loss.append(g_loss_m/i)
            self.total_d_loss.append(d_loss_m/self.d_steps)

            if step % 10 == 0:
                self.save()

            if step % 10 == 0:
                self.generator.eval()
                with torch.no_grad():
                    fake_img = self.generator(self.fixed_noise).detach().cpu()
                self.save_img(fake_img, step)
                self.generator.train()

        self.graph()
    def graph(self):
        plt.plot(self.total_g_loss, label="G_Loss")
        plt.plot(self.total_d_loss, label="D_Loss")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    def generate(self):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(1, self.latent_dim, 1, 1,device=self.device)
            img = self.generator(noise)
        self.generator.train()
        return img
    def save_img(self, img, e):
        img = img.detach().squeeze().cpu()
        img = (img + 1) / 2
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(f"generatedImages/gen{e}.png")

gan = GAN(lr_g=8e-5,lr_d=8e-5, latent_dim=100, batch_size=128, steps=460)
print(gan.device)

load = input("Wanna load model?[Y/n]: ")
if load.lower() != "n":
    gan.load()
else:
    gan.models_init()

train = input("Wanna train model?[Y/n]: ")
if train.lower() != "n":
    gan.train()

gen = input("Wanna generate sample?[Y/n]: ")
if gen.lower() != "n":
    for i in range(20):
        img = gan.generate()
        gan.save_img(img, i)
        #img = (img + 1) / 2
        #plt.imshow(img.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
        #plt.show()
