import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
from torch.nn import functional as F


class Encoder(nn.Module):
    """docstring for Encoder"""

    def __init__(self, x_dim, z_dim, n_units):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_units = n_units
        layers = [nn.Linear(x_dim, n_units[0]), nn.ELU()]
        for i in range(1, len(n_units)):
            layers.append(nn.BatchNorm1d(n_units[i - 1]))
            layers.append(nn.Linear(n_units[i - 1], n_units[i]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(n_units[-1], z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, n_units, x_type="binary"):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_units = n_units
        layers = [nn.Linear(z_dim, n_units[0]), nn.ELU()]
        for i in range(1, len(n_units)):
            layers.append(nn.BatchNorm1d(n_units[i - 1]))
            layers.append(nn.Linear(n_units[i - 1], n_units[i]))
            layers.append(nn.ELU())

        layers.append(nn.Linear(n_units[-1], x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

    def predict(self, z):
        return torch.sigmoid(self.net(z))


class AutoEncoder(nn.Module):
    def __init__(self, x_dim, z_dim, n_units, output_dist):
        super().__init__()
        if hasattr(x_dim, "__len__"):
            if len(x_dim) > 1:
                raise Exception("Multidimensional input where scalar was expected")
            x_dim = x_dim[0]

        if hasattr(z_dim, "__len__") and len(z_dim) == 1:
            if len(z_dim) > 1:
                raise Exception("Multidimensional input where scalar was expected")
            z_dim = z_dim[0]
        x_dim = int(x_dim)
        z_dim = int(z_dim)
        self.encoder = Encoder(x_dim, z_dim, n_units)
        if output_dist.lower() == "binary":
            self.decoder = Decoder(z_dim, x_dim, n_units, "binary")

    def forward(self, x):
        z = self.encoder(x)
        xcap = self.decoder(z)
        return xcap

    def predict(self, x):
        x = self.encoder(x)
        x = self.decoder.predict(x)
        return x


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

