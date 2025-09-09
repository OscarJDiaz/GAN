import argparse
import os
import random
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999
print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataroot = "./data/celeba/celebA_test"
workers = 2
batch_size = 64
image_size = 64
nc = 3       # channels
nz = 100     # latent dim
ngf = 64
ndf = 64
num_epochs = 100
# WGAN-GP typically uses Adam with these betas
lr = 0.0001
beta1 = 0.0
beta2 = 0.9
n_critic = 5
lambda_gp = 10.0
ngpu = 1

dataset = dset.ImageFolder(
    root=dataroot,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=True,
    drop_last=True,
)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Device:", device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(-1)


netG = Generator().to(device)
netC = Critic().to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netC = nn.DataParallel(netC, list(range(ngpu)))

netG.apply(weights_init)
netC.apply(weights_init)
print(netG)
print(netC)

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(beta1, beta2))

def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

img_list = []
G_losses = []
C_losses = []
iters = 0

start = time.time()
print("Start Time:", str(datetime.datetime.now().time()))

netG.train()
netC.train()

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device, non_blocking=True)
        C_mean_iteration_loss = 0.0
        for _ in range(n_critic):
            optimizerC.zero_grad(set_to_none=True)

            # Real
            output_real = netC(real_images)

            # Fake
            noise = torch.randn(real_images.size(0), nz, 1, 1, device=device)
            fake_images = netG(noise).detach()
            output_fake = netC(fake_images)

            # WGAN-GP loss
            gp = compute_gradient_penalty(netC, real_images.data, fake_images.data)
            errC = -(output_real.mean() - output_fake.mean()) + gp

            errC.backward()
            optimizerC.step()

            C_mean_iteration_loss += errC.item() / n_critic

        optimizerG.zero_grad(set_to_none=True)
        noise = torch.randn(real_images.size(0), nz, 1, 1, device=device)
        fake_images = netG(noise)
        output_fake = netC(fake_images)
        errG = -output_fake.mean()
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\t[Loss_C: %.4f] [Loss_G: %.4f]' % (
                epoch, num_epochs, i, len(dataloader), C_mean_iteration_loss, errG.item()))

        G_losses.append(errG.item())
        C_losses.append(C_mean_iteration_loss)
        iters += 1

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

finalTime = time.time() - start
print("Final Time:", str(datetime.datetime.now().time()))
print("Elapsed seconds:", finalTime)

plt.figure(figsize=(10, 5))
plt.title("Generator and Critic Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(C_losses, label="C")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

os.makedirs('./pesos/DCWGAN', exist_ok=True)

generator_path = './pesos/DCWGAN/generator_weights.pth'
discriminator_path = './pesos/DCWGAN/discriminator_weights.pth'

torch.save(netG.state_dict(), generator_path)
torch.save(netC.state_dict(), discriminator_path)
print("Saved:", generator_path, discriminator_path)

fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
try:
    HTML(ani.to_jshtml())
except Exception as e:
    print("Animation export skipped:", e)

# Show last grid
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Fake Images (last epoch)")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()

# Real vs Fake comparison
real_batch = next(iter(dataloader))
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
