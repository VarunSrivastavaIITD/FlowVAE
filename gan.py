import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import models
from torch.utils.tensorboard import SummaryWriter
from utils.load_model import save_checkpoint, load_checkpoint
import argparse
import skimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mnist_data(device, reshape=True):
    def my_transform(x):
        if reshape:
            return x.to(device).reshape(-1)
        else:
            return x.to(device)
    preprocess = transforms.Compose([transforms.ToTensor(),my_transform])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True,
    )

    return train_loader, test_loader


train_loader, test_loader = get_mnist_data(device, reshape=True)
x_dim = 784
z_dim = 20
nnum_epochs2 = 50
generator = models.Encoder(z_dim, x_dim, [100,300,300,600]).to(device)
discriminator = models.Discriminator(x_dim, [300,300,100]).to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

def disc_loss(real_z, fake_z, mode="n"):
    if mode=="n":
        real_labels = torch.rand(batch_size,1).to(device)*0.5 + 0.7
        real_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
            input=discriminator(real_z), target=real_labels
        ).sum(-1).mean()
        fake_labels = torch.rand(batch_size,1).to(device)*0.3
        fake_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
            input=discriminator(fake_z), target=fake_labels
        ).sum(-1).mean()
        return real_loss+fake_loss
    if mode=="w":
        real_loss = -discriminator(real_z).mean()
        fake_loss = discriminator(fake_z).mean()
        a = torch.rand(real_z.shape[0],1).repeat(1,real_z.shape[1]).to(real_z.device)
        z_r = a*fake_z + (1-a)*real_z
        grads = torch.autograd.grad(discriminator(z_r).sum(), z_r, create_graph=True)
        penalty = ((grads[0].reshape(real_z.shape[0],-1).norm(dim=1) - 1)**2).mean()
        return real_loss + fake_loss + 10*penalty

def gen_loss(fake_z, mode="n"):
    if mode=="n":
        gen_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
            input=discriminator(fake_z), target=torch.ones(batch_size, 1).to(device)
        ).sum(-1).mean()
    if mode=="w":
        gen_loss = -discriminator(fake_z).mean()
    return gen_loss

for e in range(1, num_epochs2 + 1):
    ## Train discriminator on real z's and fake z's
    ## min -log(G(real)) - log(1-G(fake))
    for batch in train_loader:
        x_batch, y_batch = batch
        batch_size = x_batch.size()[0]
        labels = torch.eye(10)[y_batch.cpu()].to(device).float()
        generator.train()
        discriminator.train()
        d_optimizer.zero_grad()
        real_z = x_batch
        fake_z = generator(torch.randn(batch_size, z_dim).to(device))
        d_loss = disc_loss(real_z,fake_z,mode="w")
        d_loss.backward()
        d_optimizer.step()
    ## Train generator
    ## min -log(G(fake))
    fake_z = generator(torch.randn(batch_size, z_dim).to(device))
    g_loss = gen_loss(fake_z, mode="w")
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    print(
        "Epoch {} : Disc loss = {:.2e} Gen loss = {:.2e}".format(
            e, d_loss, g_loss
        )
    )
    generator.eval()
        with torch.no_grad():
            z = generator(torch.randn(100, z_dim).to(device))
            images = z_dim
            images_tiled = np.reshape(
            np.transpose(np.reshape(images.cpu().detach(), (10, 10, 28, 28)), (0, 2, 1, 3)),
            (280, 280),
        )
        plt.imsave("mnist-gen/w{}.png".format(e), images_tiled, cmap="gray")