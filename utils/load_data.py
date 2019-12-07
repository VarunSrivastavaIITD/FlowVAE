from __future__ import print_function

import torch
import torch.utils.data as data_utils
import pickle
from scipy.io import loadmat
import torchvision as tv
import torch.utils.data as utdata
from sklearn.model_selection import train_test_split

import numpy as np

import os


def load_mnist(args, **kwargs):
    """
    Dataloading function for mnist. Outputs image data in vectorized form: each image is a vector of size 784
    """
    args.dynamic_binarization = False
    args.input_type = "binary"

    flatten = kwargs.get("flatten", False)

    # start processing
    transforms_list = [
        tv.transforms.ToTensor(),
        # tv.transforms.Normalize((0.5,), (0.5,)),
    ]
    args.xdim = (28, 28)
    if flatten:
        transforms_list.append(tv.transforms.Lambda(lambda x: x.view(-1)))
        args.xdim = (784,)
    preprocess = tv.transforms.Compose(transforms_list)
    preprocess = kwargs.get("preprocess", preprocess)

    train_dataset = tv.datasets.MNIST(
        "./data", transform=preprocess, download=True, train=True
    )
    train_len = len(train_dataset)

    if args.val_frac:
        train_indices, validation_indices = train_test_split(
            np.arange(train_len), test_size=args.val_frac, random_state=args.manual_seed
        )
        train_indices = train_indices.tolist()
        validation_indices = validation_indices.tolist()
    else:
        train_indices = np.arange(train_len).tolist()
        val_indices = []

    train = utdata.Subset(train_dataset, train_indices)
    validation = utdata.Subset(train_dataset, validation_indices)

    # pytorch data loader
    train_loader = data_utils.DataLoader(
        train, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory
    )

    val_loader = data_utils.DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
    )

    test = tv.datasets.MNIST("./data", transform=preprocess, download=True, train=False)
    test_loader = data_utils.DataLoader(
        test, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory
    )

    return train_loader, val_loader, test_loader, args


def load_cifar10(args, **kwargs):
    flatten = kwargs.get("flatten", False)

    # start processing
    transforms_list = [
        tv.transforms.ToTensor(),
        # tv.transforms.Normalize((0.5,), (0.5,)),
    ]
    args.xdim = (3, 32, 32)
    if flatten:
        transforms_list.append(tv.transforms.Lambda(lambda x: x.view(-1)))
        args.xdim = (1024,)
    preprocess = tv.transforms.Compose(transforms_list)
    preprocess = kwargs.get("preprocess", preprocess)

    train_dataset = tv.datasets.CIFAR10(
        "./data", transform=preprocess, download=True, train=True
    )
    train_len = len(train_dataset)

    if args.val_frac:
        train_indices, validation_indices = train_test_split(
            np.arange(train_len), test_size=args.val_frac, random_state=args.manual_seed
        )
        train_indices = train_indices.tolist()
        validation_indices = validation_indices.tolist()
    else:
        train_indices = np.arange(train_len).tolist()
        val_indices = []

    train = utdata.Subset(train_dataset, train_indices)
    validation = utdata.Subset(train_dataset, validation_indices)

    # pytorch data loader
    train_loader = data_utils.DataLoader(
        train, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory
    )

    val_loader = data_utils.DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
    )

    test = tv.datasets.CIFAR10(
        "./data", transform=preprocess, download=True, train=False
    )
    test_loader = data_utils.DataLoader(
        test, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory
    )

    return train_loader, val_loader, test_loader, args


def load_dataset(args, **kwargs):

    if args.dataset == "mnist":
        train_loader, val_loader, test_loader, args = load_mnist(args, **kwargs)
    elif args.dataset == "cifar10":
        train_loader, val_loader, test_loader, args = load_cifar10(args, **kwargs)
    else:
        raise Exception("Wrong name of the dataset!")

    return train_loader, val_loader, test_loader, args
