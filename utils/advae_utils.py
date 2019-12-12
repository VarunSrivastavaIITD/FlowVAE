import numpy as np
import torch
from torch import optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import skimage.io

def get_data(dataset, device, reshape=True, batch_size=100):
    def my_transform(x):
        if reshape:
            return x.to(device).reshape(-1)
        else:
            return x.to(device)

    preprocess = transforms.Compose([transforms.ToTensor(), my_transform])
    if dataset=='svhn':
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN("data/"+dataset, split='train', download=True, transform=preprocess),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN("data/"+dataset, split='test', download=True, transform=preprocess),
            batch_size=batch_size,
            shuffle=True,
        )
        return train_loader, test_loader
    if dataset=='mnist': DS = datasets.MNIST
    if dataset=='fmnist': DS = datasets.FashionMNIST
    if dataset=='cifar10': DS = datasets.CIFAR10
    train_loader = torch.utils.data.DataLoader(
        DS("data/"+dataset, train=True, download=True, transform=preprocess),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        DS("data/"+dataset, train=False, download=True, transform=preprocess),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, test_loader

def noisy_soft_labels(labels):
    noisy = torch.bernoulli(0.9 * labels + 0.05)
    noisy_soft = torch.where(
        noisy == 0, torch.rand_like(noisy) * 0.3, torch.rand_like(noisy) * 0.7 + 0.5
    ).to(labels.device)
    return noisy_soft

def disc_loss(real_z, fake_z, discriminator, mode="n"):
    if mode[-1]=="n":
        batch_size = real_z.shape[0]
        device = real_z.device
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
        a = torch.rand(real_z.shape[0], *[1]*n_dims).repeat(1, *real_z.shape[1:]).to(real_z.device)
        z_r = a * fake_z + (1 - a) * real_z
        grads = torch.autograd.grad(discriminator(z_r).sum(), z_r, create_graph=True)
        penalty = ((grads[0].reshape(real_z.shape[0], -1).norm(dim=1) - 1) ** 2).mean()
        return real_loss + fake_loss + 10 * penalty


def gen_loss(fake_z, discriminator, mode="n"):
    if mode[-1]=="n":
        batch_size = real_z.shape[0]
        gen_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
            input=discriminator(fake_z), target=torch.ones(batch_size, 1).to(real_z.device)
        ).sum(-1).mean()
    if mode[-1]=="w":
        gen_loss = -discriminator(fake_z).mean()
    return gen_loss

def save_tiled_images(images, filename, dataset):
    if dataset in ['mnist','fmnist']:
        images_tiled = np.reshape(np.transpose(np.reshape(images, (10, 10, 28, 28)), (0, 2, 1, 3)), (280, 280))
        plt.imsave(filename, images_tiled, cmap="gray")
    if dataset in ['cifar10','svhn']:
        images_tiled = np.reshape(np.transpose(np.reshape(images, (10,10,3,32,32)), (0,3,1,4,2)), (320,320,3))
        plt.imsave(filename, images_tiled)