import numpy as np
import torch
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
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help="Which dataset?")
parser.add_argument('--ae_mode', default='ae', help="Autoencoder mode : ae or conv")
parser.add_argument('--train_ae', type=int, default=20, help="Number of epochs to train autoencoder")
parser.add_argument('--ae_epoch', type=int, default=5, help="Epoch of AE model to load")
parser.add_argument('--gen_mode', default='w', help="Generator mode : n, w")
parser.add_argument('--train_gen', type=int, default=40, help="Number of epochs to train generator")
parser.add_argument('--gen_epoch', type=int, default=40, help="Epoch of gen model")
parser.add_argument('--save_images', type=int, default=0, help="Save generated images")
parser.add_argument('--save_latents', type=int, default=0, help="Save generated latent vectors")
parser.add_argument('--classify', type=int, default=0, help="Train classifier")
args = parser.parse_args()

writer = SummaryWriter()
num_epochs1 = args.train_ae
num_epochs2 = args.train_gen
train_loader, test_loader = get_data(args.dataset, device, reshape=(not args.ae_mode=="conv"))
save_path = 'checkpoints/'+args.dataset
if args.dataset=='mnist':
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    sample_fn = lambda x:torch.round(torch.sigmoid(x))
    x_dim = 784
if args.dataset in ['fmnist','svhn']:
    loss_fn = torch.nn.MSELoss(reduction='none')
    sample_fn = lambda x:torch.sigmoid(x)
if args.dataset in ['fmnist','mnist']:
    x_dim = 784
    z_dim = 64
    z0_dim = 64
if args.dataset=='svhn':
    x_dim = 3072
    z_dim = z0_dim = 50

if args.ae_mode=="ae" or args.ae_mode=="sup":
    ae = models.AutoEncoder(x_dim, z_dim, n_units=[500,500], sup=(args.ae_mode=="sup")).to(device)
    generator = models.Encoder(z0_dim, z_dim, [500,500]).to(device)
    discriminator = models.Discriminator(z_dim, [20,20]).to(device)
if args.ae_mode=="conv":
    ae = models.ConvAutoEncoder(in_channels=1, image_size=(28,28), activation=None).to(device)
    generator = models.ConvGenerator(z0_dim, ae.z_dim).to(device)
    discriminator = models.ConvDiscriminator(ae.z_dim).to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=1e-2)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
ed_optimizer = optim.Adam(ae.parameters(), lr=1e-3)

if args.train_ae:
    # First train encoder and decoder
    print("Training Encoder-Decoder .............")
    for e in range(1, num_epochs1 + 1):
        for batch in train_loader:
            ae.train()
            x_batch, y_batch = batch
            batch_size = x_batch.size()[0]
            y_hot = torch.eye(10)[y_batch.cpu()].to(device).float()

            ## Train encoder-decoder
            ## min -E_{q(z|x)} log(p(x|z))
            ed_optimizer.zero_grad()
            if args.ae_mode=="sup": z = ae.encode(torch.cat((x_batch,y_hot),-1))
            else : z = ae.encode(x_batch)
#             z += torch.randn_like(z)*0.2
            if args.ae_mode=="sup": z = torch.cat((z,y_hot),-1)
            x_out = ae.decode(z)
            ed_loss = (
                loss_fn(
                    input=x_out, target=x_batch
                )
                .sum(-1)
                .mean()
            )
            ed_loss.backward()
            ed_optimizer.step()

        with torch.no_grad():
            x_batch, y_batch = next(iter(test_loader))
            y_hot = torch.eye(10)[y_batch.cpu()].to(device).float()
            ae.eval()
            if args.ae_mode=="sup": z = ae.encode(torch.cat((x_batch,y_hot),-1))
            else : z = ae.encode(x_batch)
            if args.ae_mode=="sup": z = torch.cat((z,y_hot),-1)
            x_out = ae.decode(z)
            images = sample_fn(x_out).cpu().detach()
            test_loss = (
                loss_fn(
                    input=x_out, target=x_batch
                )
                .sum(-1)
                .mean()
            )
        save_tiled_images(images, "images/{}-ae/{}{}.png".format(args.dataset,args.ae_mode,e), args.dataset)
        print(
            "Epoch {} : E-D train loss = {:.2e} test loss = {:.2e}".format(
                e, ed_loss, test_loss
            )
        )

        if e % 5 == 0:
            checkpoint_dict = {
                "epoch": e,
                "autoencoder": ae.state_dict(),
                "ed_optimizer": ed_optimizer.state_dict(),
            }
            fname = f'enc-dec_{args.ae_mode}{e}'
            save_checkpoint(checkpoint_dict, save_path, fname)
else:
    fname = 'enc-dec_'+args.ae_mode+str(args.ae_epoch)
    enc_dec = load_checkpoint(save_path, fname, device)
    ae.load_state_dict(enc_dec["autoencoder"])

if args.train_gen:
    ae.eval()
    print("Training Discriminator and generator")
    for e in range(1, num_epochs2 + 1):
        ## Train discriminator on real z's and fake z's
        ## min -log(G(real)) - log(1-G(fake))
        avg_d_loss = 0
        count = 0
        for batch in train_loader:
            x_batch, y_batch = batch
            batch_size = x_batch.size()[0]
            y_hot = torch.eye(10)[y_batch.cpu()].to(device).float()
            generator.train()
            discriminator.train()
            d_optimizer.zero_grad()
            if args.ae_mode=="sup": real_z = ae.encode(torch.cat((x_batch,y_hot),-1))
            else : real_z = ae.encode(x_batch)
            fake_z = generator(torch.randn(batch_size, z0_dim).to(device))
            d_loss = disc_loss(real_z, fake_z, discriminator, mode=args.gen_mode)
            d_loss.backward()
            d_optimizer.step()
            avg_d_loss += d_loss
            count += 1
        avg_d_loss /= count
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
        writer.add_scalar('gen_loss', g_loss, e)
        writer.add_scalar('disc_loss', avg_d_loss, e)

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
            if args.ae_mode=="sup":
                z = generator(torch.randn(10, z0_dim).to(device)).repeat(10,1)
                y = torch.eye(10).repeat_interleave(10,0).to(device)
                z = torch.cat((z,y),-1)
            images = sample_fn(ae.decode(z)).cpu().detach()
        save_tiled_images(images, "images/{}-gen/{}{}{}.png".format(args.dataset,args.ae_mode,args.gen_mode,e), args.dataset)

        if e % 10 == 0:
            checkpoint_dict = {
                "epoch": e,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
            }
            fname = f'gen-disc_{args.ae_mode}{args.gen_mode}{e}'
            save_checkpoint(checkpoint_dict, save_path, fname)
else:
#     print("Nothing")
    fname = f'gen-disc_{args.ae_mode}{args.gen_mode}{args.gen_epoch}'
    enc_dec = load_checkpoint(save_path, fname, device)
    generator.load_state_dict(enc_dec["generator"])

if args.save_images:
    with torch.no_grad():
        if args.ae_mode=="sup":
#             z1, z2 = torch.meshgrid(torch.linspace(-1,1,10), torch.linspace(-1,1,10))
#             z = torch.cat((z1.reshape(10,10,1),z2.reshape(10,10,1)),2).reshape(100,2).to(device)
            z = generator(torch.randn(100, z0_dim).to(device))
            for y in range(10):
                y_hot = torch.eye(10)[y].reshape(1,10).repeat(100,1).to(device)
                zcat = torch.cat((z,y_hot),-1)
                images = sample_fn(ae.decode(zcat)).cpu().detach()
                save_tiled_images(images, "images/{}-sup{}.png".format(args.dataset,y), args.dataset)
        else:
            z = generator(torch.randn(10000, z_dim).to(device))
            images = sample_fn(ae.decode(z))
            images = images.cpu().detach().numpy() * 255
            images = np.reshape(images, (-1, 28, 28)).astype(np.uint8)
            for i in range(10000):
                skimage.io.imsave(f'images/advae-{args.ae_mode}{args.gen_mode}-samples/{i}.png', images[i])

if args.save_latents:
    _,test_loader = get_data(args.dataset, device, reshape=(args.ae_mode=="ae"), batch_size=10000)
    x_batch, y_batch = next(iter(test_loader))
    z = ae.encode(x_batch).reshape(10000,-1)
    np.savez(f'images/advae-{args.ae_mode}{args.gen_mode}-latents.npz', X=z.cpu().detach(), y=y_batch.cpu().detach())
            
if args.classify:
    print("Training classifier .....")
    classifier = models.Classifier(z_dim=z_dim, n_cat=10, n_units=[300,300]).to(device)
    c_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    num_epochs3 = 20
    for e in range(1, num_epochs3 + 1):
        for batch in train_loader:
            classifier.train()
            x_batch, y_batch = batch
            y_batch = y_batch.to(device)
            batch_size = x_batch.size()[0]
            
            z_batch = ae.encode(x_batch)
            scores = classifier(z_batch)
            c_loss = torch.nn.CrossEntropyLoss(reduction='mean')(scores, y_batch)
            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()
            
        with torch.no_grad():
            x_batch, y_batch = next(iter(test_loader))
            y_batch = y_batch.to(device)
            batch_size = x_batch.size()[0]
            classifier.eval()
            z = ae.encode(x_batch)
            test_loss = torch.nn.CrossEntropyLoss(reduction='mean')(scores, y_batch)
            preds = classifier.predict(z)
            accuracy = torch.sum((preds==y_batch).float())*100/batch_size
        print(
            "Epoch {} : Classifier train loss = {:.2e} test loss = {:.2e} test accuracy = {:.2e}%".format(
                e, c_loss, test_loss, accuracy
            )
        )
    checkpoint_dict = {
        'epoch':e,
        'classifier':classifier.state_dict(),
        'c_optimizer':c_optimizer.state_dict()
    }
    fname = f'classifier_{args.ae_mode}'
    save_checkpoint(checkpoint_dict, save_path, fname)