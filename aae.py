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
import argparse
import skimage.io

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

def noisy_soft_labels(labels):
    noisy = torch.bernoulli(0.9*labels+0.05)
    noisy_soft = torch.where(noisy==0, torch.rand_like(noisy)*0.3, torch.rand_like(noisy)*0.7+0.5)
    return noisy_soft

parser = argparse.ArgumentParser()
parser.add_argument('--ae_mode', default='ae', help="Autoencoder mode : ae or conv")
parser.add_argument('--train_ae', type=int, default=10, help="Number of epochs to train autoencoder")
parser.add_argument('--ae_epoch', type=int, default=10, help="Epoch of AE model to load")
parser.add_argument('--gen_mode', default='n', help="Generator mode : n, w")
parser.add_argument('--save_images', type=int, default=0, help="Save generated images")
parser.add_argument('--classify', type=int, default=0, help="Train classifier")
args = parser.parse_args()

train_loader, test_loader = get_mnist_data(device, reshape=(args.ae_mode=="ae"))
num_epochs1 = args.train_ae
x_dim = 784
z_dim = 10
z0_dim = 10
save_path = 'checkpoints_aae'

if args.ae_mode=="ae":
    ae = models.AutoEncoder(x_dim, z_dim, n_units=[300,300]).to(device)
if args.ae_mode=="conv":
    ae = models.ConvAutoEncoder(in_channels=1, image_size=(28,28), z_dim=z_dim, activation=None).to(device)
discriminator = models.Discriminator(z_dim, [20,20]).to(device)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
ed_optimizer = optim.Adam(ae.parameters(), lr=1e-3)

if args.train_ae:
# First train encoder and decoder
    print("Training Encoder-Decoder and Discriminator jointly .............")
    for e in range(1, num_epochs1 + 1):
        for batch in train_loader:
            ae.train()
            x_batch, y_batch = batch
            batch_size = x_batch.size()[0]
            labels = torch.eye(10)[y_batch.cpu()].to(device).float()

            ## Train encoder-decoder
            ## min -E_{q(z|x)} log(p(x|z))
            ed_optimizer.zero_grad()
            z = ae.encode(x_batch)
            z += torch.randn_like(z)*0.2
            x_out = ae.decode(z)
            ed_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(input=x_out, target=x_batch).sum(-1).mean()
            ed_loss.backward()
            ed_optimizer.step()
            ## Train discriminator on real z's and fake z's
            ## min -log(G(real)) - log(1-G(fake))
            discriminator.train()
            d_optimizer.zero_grad()
            fake_z = ae.encode(x_batch).detach()
            real_z = torch.randn(batch_size, z0_dim).to(device)
            d_loss = disc_loss(real_z, fake_z, mode=args.gen_mode)
            d_loss.backward()
            d_optimizer.step()
            ## Train generator (decoder)
            ## min -log(G(fake))
            fake_z = ae.encode(x_batch)
            g_loss = gen_loss(fake_z, mode=args.gen_mode)
            ed_optimizer.zero_grad()
            g_loss.backward()
            ed_optimizer.step()

        with torch.no_grad():
            ae.eval()
            z = torch.randn(batch_size, z0_dim).to(device)
            x_out = ae.decode(z)
            images = torch.round(torch.sigmoid(x_out).cpu().detach())
        images_tiled = np.reshape(
            np.transpose(np.reshape(images, (10, 10, 28, 28)), (0, 2, 1, 3)),
            (280, 280),
        )
        plt.imsave("images/mnist-aae/{}{}{}.png".format(args.ae_mode,args.gen_mode,e), images_tiled, cmap="gray")
        print(
            "Epoch {} : E-D train loss = {:.2e} Disc loss = {:.2e} Gen loss = {:.2e}".format(
                e, ed_loss, d_loss, g_loss
            )
        )
        
        if e%5==0:
            checkpoint_dict = {
                'epoch':e,
                'autoencoder':ae.state_dict(),
                'ed_optimizer':ed_optimizer.state_dict()
            }
            fname = f'enc-dec_{args.ae_mode}{e}'
            save_checkpoint(checkpoint_dict, save_path, fname)

else:
    fname = 'enc-dec_'+args.ae_mode+str(args.ae_epoch)
    enc_dec = load_checkpoint(save_path, fname, device)
    ae.load_state_dict(enc_dec['autoencoder'])

def gen_loss(fake_z, mode="n"):
    if mode[-1]=="n":
        gen_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
            input=discriminator(fake_z), target=torch.ones(batch_size, 1).to(device)
        ).sum(-1).mean()
    if mode[-1]=="w":
        gen_loss = -discriminator(fake_z).mean()
    return gen_loss
    
def disc_loss(real_z, fake_z, mode="n"):
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
        a = torch.rand(real_z.shape[0],1).repeat(1,real_z.shape[1]).to(real_z.device)
        z_r = a*fake_z + (1-a)*real_z
        grads = torch.autograd.grad(discriminator(z_r).sum(), z_r, create_graph=True)
        penalty = ((grads[0].reshape(real_z.shape[0],-1).norm(dim=1) - 1)**2).mean()
        return real_loss + fake_loss + 10*penalty

if args.save_images:
    with torch.no_grad():
        z = torch.randn(10000, z_dim).to(device)
        images = torch.round(torch.sigmoid(ae.decode(z)))
        images = images.cpu().detach().numpy()*255
        images = np.reshape(images,(-1,28,28)).astype(np.uint8)

        for i in range(10000):
            skimage.io.imsave(f'images/aae-{args.ae_mode}{args.gen_mode}-samples/{i}.png', images[i])
            
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