import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
from torch.nn import functional as F

class Classifier(nn.Module):
    def __init__(self, z_dim, n_cat, n_units):
        super().__init__()
        self.z_dim = z_dim
        layers = [nn.Linear(z_dim, n_units[0]), nn.BatchNorm1d(n_units[0]), nn.ELU()]
        for i in range(1, len(n_units)):
            layers.append(nn.Linear(n_units[i - 1], n_units[i]))
            layers.append(nn.BatchNorm1d(n_units[i - 1]))            
            layers.append(nn.ELU())
        layers.append(nn.Linear(n_units[-1], n_cat))
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)
    def predict(self, z):
        return self.net(z).argmax(-1)