import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import models

# from torch.utils.tensorboard import SummaryWriter
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

    preprocess = transforms.Compose([transforms.ToTensor(), my_transform])
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
    noisy = torch.bernoulli(0.9 * labels + 0.05)
    noisy_soft = torch.where(
        noisy == 0, torch.rand_like(noisy) * 0.3, torch.rand_like(noisy) * 0.7 + 0.5
    )
    return noisy_soft


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_ae", type=int, default=10, help="Number of epochs to train autoencoder"
)
parser.add_argument(
    "--ae_epoch", type=int, default=10, help="Epoch of AE model to load"
)
parser.add_argument(
    "--train_gen", type=int, default=50, help="Number of epochs to train generator"
)
parser.add_argument("--gen_epoch", type=int, default=40, help="Epoch of gen model")
parser.add_argument("--save_images", type=int, default=0, help="Save generated images")
args = parser.parse_args()

train_loader, test_loader = get_mnist_data(device, reshape=False)
num_epochs1 = args.train_ae
num_epochs2 = args.train_gen
x_dim = 784
z_dim = 392
z0_dim = 392
save_path = "checkpoints/advae"

ae = models.ConvAutoEncoder(in_channels=1, image_size=(28, 28), activation=None).to(
    device
)
# ae = models.AutoEncoder(x_dim, z_dim, n_units=[300,300]).to(device)
# generator = models.Encoder(z0_dim, z_dim, [300,300]).to(device)
generator = models.ConvGenerator(z0_dim, ae.z_dim)
discriminator = models.Discriminator(z_dim, [20, 20]).to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=1e-2)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
ed_optimizer = optim.Adam(ae.parameters(), lr=1e-3)

# writer = SummaryWriter()
if args.train_ae:
    # First train encoder and decoder
    print("Training Encoder-Decoder .............")
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
            x_out = ae.decode(z)
            ed_loss = (
                torch.nn.BCEWithLogitsLoss(reduction="none")(
                    input=x_out, target=x_batch
                )
                .sum(-1)
                .mean()
            )
            ed_loss.backward()
            ed_optimizer.step()

        with torch.no_grad():
            x_batch = next(iter(test_loader))[0]
            ae.eval()
            z = ae.encode(x_batch)
            x_out = ae.decode(z)
            images = torch.round(torch.sigmoid(x_out).cpu().detach())
            test_loss = (
                torch.nn.BCEWithLogitsLoss(reduction="none")(
                    input=x_out, target=x_batch
                )
                .sum(-1)
                .mean()
            )
        images_tiled = np.reshape(
            np.transpose(np.reshape(images, (10, 10, 28, 28)), (0, 2, 1, 3)),
            (280, 280),
        )
        plt.imsave("mnist-ae/conv{}.png".format(e), images_tiled, cmap="gray")
        print(
            "Epoch {} : E-D train loss = {:.2e} test loss = {:.2e}".format(
                e, ed_loss, test_loss
            )
        )
        # writer.add_scalars('losses', {'train':ed_loss, 'test':test_loss}, e)

        if e % 5 == 0:
            checkpoint_dict = {
                "epoch": e,
                "autoencoder": ae.state_dict(),
                "ed_optimizer": ed_optimizer.state_dict(),
            }
            fname = f"enc-dec_{e}"
            save_checkpoint(checkpoint_dict, save_path, fname)
else:
    fname = "enc-dec_" + str(args.ae_epoch)
    enc_dec = load_checkpoint(save_path, fname, device)
    ae.load_state_dict(enc_dec["autoencoder"])


def disc_loss(real_z, fake_z, mode="n"):
    if mode == "n":
        real_labels = torch.rand(batch_size, 1).to(device) * 0.5 + 0.7
        real_loss = (
            torch.nn.BCEWithLogitsLoss(reduction="none")(
                input=discriminator(real_z), target=real_labels
            )
            .sum(-1)
            .mean()
        )
        fake_labels = torch.rand(batch_size, 1).to(device) * 0.3
        fake_loss = (
            torch.nn.BCEWithLogitsLoss(reduction="none")(
                input=discriminator(fake_z), target=fake_labels
            )
            .sum(-1)
            .mean()
        )
        return real_loss + fake_loss
    if mode == "w":
        real_loss = -discriminator(real_z).mean()
        fake_loss = discriminator(fake_z).mean()
        a = torch.rand(real_z.shape[0], 1).repeat(1, real_z.shape[1]).to(real_z.device)
        z_r = a * fake_z + (1 - a) * real_z
        grads = torch.autograd.grad(discriminator(z_r).sum(), z_r, create_graph=True)
        penalty = ((grads[0].reshape(real_z.shape[0], -1).norm(dim=1) - 1) ** 2).mean()
        return real_loss + fake_loss + 10 * penalty


def gen_loss(fake_z, mode="n"):
    if mode == "n":
        gen_loss = (
            torch.nn.BCEWithLogitsLoss(reduction="none")(
                input=discriminator(fake_z), target=torch.ones(batch_size, 1).to(device)
            )
            .sum(-1)
            .mean()
        )
    if mode == "w":
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
            d_loss = disc_loss(real_z, fake_z, mode="w")
            d_loss.backward()
            d_optimizer.step()
        ## Train generator
        ## min -log(G(fake))
        fake_z = generator(torch.randn(batch_size, z0_dim).to(device))
        g_loss = gen_loss(fake_z, mode="w")
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
            images = torch.round(torch.sigmoid(ae.decode(z)))
        images_tiled = np.reshape(
            np.transpose(
                np.reshape(images.cpu().detach(), (10, 10, 28, 28)), (0, 2, 1, 3)
            ),
            (280, 280),
        )
        plt.imsave("mnist-gen/convw{}.png".format(e), images_tiled, cmap="gray")

        if e % 10 == 0:
            checkpoint_dict = {
                "epoch": e,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
            }
            fname = f"gen-disc_{e}"
            save_checkpoint(checkpoint_dict, save_path, fname)
else:
    fname = f"gen-disc_{args.gen_epoch}"
    enc_dec = load_checkpoint(save_path, fname, device)
    generator.load_state_dict(enc_dec["generator"])

if args.save_images:
    with torch.no_grad():
        z = generator(torch.randn(10000, z_dim).to(device))
        images = torch.round(torch.sigmoid(ae.decode(z)))
        images = images.cpu().detach().numpy() * 255
        images = np.reshape(images, (-1, 28, 28)).astype(np.uint8)

        for i in range(10000):
            skimage.io.imsave(f"advae-samples/{i}.png", images[i])
