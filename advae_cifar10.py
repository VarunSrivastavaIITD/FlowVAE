import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import models
from utils.load_model import save_checkpoint, load_checkpoint
from utils.advae_utils import *
import argparse
import skimage.io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help="Which dataset?")
parser.add_argument('--train_ae', type=int, default=20, help="Number of epochs to train autoencoder")
parser.add_argument('--ae_epoch', type=int, default=20, help="Epoch of AE model to load")
parser.add_argument('--gen_mode', default='w', help="Generator mode : n, w")
parser.add_argument('--train_gen', type=int, default=40, help="Number of epochs to train generator")
parser.add_argument('--gen_epoch', type=int, default=40, help="Epoch of gen model")
parser.add_argument('--save_images', type=int, default=0, help="Save generated images")
parser.add_argument('--classify', type=int, default=0, help="Train classifier")
args = parser.parse_args()

train_loader, test_loader = get_data(args.dataset, device, reshape=False)
num_epochs1 = args.train_ae
num_epochs2 = args.train_gen
z0_dim = 64
save_path = 'checkpoints/'+args.dataset

ae = models.ConvAutoEncoder(image_size=(32,32), activation=nn.Sigmoid()).to(device)
generator = models.ConvGenerator(z0_dim, ae.z_dim).to(device)
discriminator = models.ConvDiscriminator(ae.z_dim).to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
ae_optimizer = optim.Adam(ae.parameters(), lr=1e-3)

if args.train_ae:
#     First train encoder and decoder
    print("Training Encoder-Decoder .............")
    for e in range(1, num_epochs1 + 1):
        for batch in train_loader:
            ae.train()
            x_batch, y_batch = batch
            batch_size = x_batch.size()[0]
            labels = torch.eye(10)[y_batch.cpu()].to(device).float()

            ## Train encoder-decoder
            ## min -E_{q(z|x)} log(p(x|z))
            ae_optimizer.zero_grad()
            z = ae.encode(x_batch)
            z += torch.randn_like(z)*0.2
            x_out = ae.decode(z)
            ae_loss = torch.nn.MSELoss(reduction='none')(input=x_out, target=x_batch).sum(-1).mean()
            ae_loss.backward()
            ae_optimizer.step()

        with torch.no_grad():
            x_batch = next(iter(test_loader))[0]
            ae.eval()
            z = ae.encode(x_batch)
            z += torch.randn_like(z)*0.2
            x_out = ae.decode(z)
            test_loss = torch.nn.MSELoss(reduction='none')(input=x_out, target=x_batch).sum(-1).mean()
        images = x_out.cpu().detach().numpy()
        images_tiled = np.reshape(np.transpose(np.reshape(images, (10,10,3,32,32)), (0,3,1,4,2)), (320,320,3))
        plt.imsave("images/cifar-ae/{}.png".format(e), images_tiled)
        print(
            "Epoch {} : E-D train loss = {:.2e} test loss = {:.2e}".format(
                e, ae_loss, test_loss
            )
        )
        if e%5==0:
            checkpoint_dict = {'epoch':e, 'model':ae.state_dict(), 'optimizer':ae_optimizer.state_dict()}
            fname = f'conv-ae_{e}'
            save_checkpoint(checkpoint_dict, save_path, fname)
else:
    fname = 'conv-ae_'+str(args.ae_epoch)
    enc_dec = load_checkpoint(save_path, fname, device)
    ae.load_state_dict(enc_dec['model'])



def disc_loss(real_z, fake_z, discriminator, mode="n"):
    if mode[-1]=="n":
        real_labels = torch.rand(batch_size,1).to(device)*0.5 + 0.7
        real_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
            input=discriminator(real_z), target=real_labels
        ).sum(-1).mean()
        fake_labels = torch.rand(batch_size,1).to(device)*0.3
        fake_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
            input=discriminator(fake_z), target=fake_labels
        ).sum(-1).mean()
        return real_loss+fake_loss
    if mode[-1]=="w":
        real_loss = -discriminator(real_z).mean()
        fake_loss = discriminator(fake_z).mean()
        n_dims = len(real_z.shape)-1
        a = torch.rand(real_z.shape[0],*[1]*n_dims).repeat(1,*real_z.shape[1:]).to(real_z.device)
        z_r = a * fake_z + (1 - a) * real_z
        grads = torch.autograd.grad(discriminator(z_r).sum(), z_r, create_graph=True)
        penalty = ((grads[0].reshape(real_z.shape[0], -1).norm(dim=1) - 1) ** 2).mean()
        return real_loss + fake_loss + 10 * penalty


def gen_loss(fake_z, discriminator, mode="n"):
    if mode[-1]=="n":
        gen_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
            input=discriminator(fake_z), target=torch.ones(batch_size, 1).to(device)
        ).sum(-1).mean()
    if mode[-1]=="w":
        gen_loss = -discriminator(fake_z).mean()
    return gen_loss


if args.train_gen:
    ae.eval()
    print("Training Discriminator and generator")
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
            real_z = ae.encode(x_batch).detach()
            fake_z = generator(torch.randn(batch_size, z0_dim).to(device))
            d_loss = disc_loss(real_z, fake_z, discriminator, mode=args.gen_mode)
            d_loss.backward()
            d_optimizer.step()
        ## Train generator
        ## min -log(G(fake))
        fake_z = generator(torch.randn(batch_size, z0_dim).to(device))
        g_loss = gen_loss(fake_z, discriminator, mode=args.gen_mode)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        print(
            "Epoch {} : Disc loss = {:.2e} Gen loss = {:.2e}".format(e, d_loss, g_loss)
        )

        disc_grad_norm = 0
        for p in discriminator.parameters():
            param_norm = p.grad.data.norm(2)
            disc_grad_norm += param_norm.item() ** 2
        gen_grad_norm = 0
        for p in generator.parameters():
            param_norm = p.grad.data.norm(2)
            gen_grad_norm += param_norm.item() ** 2
        print(f"Disc grad norm {disc_grad_norm}")
        print(f"Gen grad norm {gen_grad_norm}")

        generator.eval()
        with torch.no_grad():
            z = generator(torch.randn(100, z0_dim).to(device))
            images = ae.decode(z).cpu().detach().numpy()
        images_tiled = np.reshape(np.transpose(np.reshape(images, (10,10,3,32,32)), (0,3,1,4,2)), (320,320,3))
        plt.imsave("images/cifar-gen/{}.png".format(e), images_tiled)

        if e % 10 == 0:
            checkpoint_dict = {
                "epoch": e,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
            }
            fname = f'gen-disc_conv{e}'
            save_checkpoint(checkpoint_dict, save_path, fname)
else:
    fname = f'gen-disc_conv{args.gen_epoch}'
    enc_dec = load_checkpoint(save_path, fname, device)
    generator.load_state_dict(enc_dec["generator"])