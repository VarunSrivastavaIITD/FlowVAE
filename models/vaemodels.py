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
    def __init__(self, x_dim, z_dim, n_units, output_dist="binary"):
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

    def encode(self,x):
        return self.encoder(x)

    def decode(self,z):
        return self.decoder(z)


class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels=3, image_size=(32,32), activation=nn.Sigmoid()):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=1),  # b, 16, 16, 16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=16),
            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=16)
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1)  # b, 8, 8, 8
            # nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),  # b, 16, 16, 16
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=16),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),  # b, 16, 32, 32
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=in_channels, kernel_size=1, stride=1),  # b, 3, 32, 32
        )
        if activation!=None:
            self.decoder.add_module("activation", activation)
        self.z_dim = (8,(image_size[0]//4),(image_size[1]//4))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self,x):
        z = self.encoder(x)
        return z.view(-1,np.prod(self.z_dim))

    def decode(self,z):
        x = self.decoder(z.view(-1,*self.z_dim))
        return x


class Discriminator(nn.Module):
    def __init__(self, z_dim, n_units):
        super().__init__()
        self.z_dim = z_dim
        self.n_units = n_units
        layers = [nn.Linear(z_dim, n_units[0]), nn.ELU()]
        for i in range(1, len(n_units)):
            layers.append(nn.Linear(n_units[i - 1], n_units[i]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(n_units[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)