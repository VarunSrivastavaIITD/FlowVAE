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
		for i in range(1,len(n_units)):
			layers.append(nn.Linear(n_units[i-1],n_units[i]))
			layers.append(nn.ELU())
		layers.append(nn.Linear(n_units[-1],z_dim))
		self.net = nn.Sequential(*layers)

	def forward(self,x):
		return self.net(x)

class Decoder(nn.Module):
	def __init__(self, z_dim, x_dim, n_units, x_type='binary'):
		super().__init__()
		self.x_dim = x_dim
		self.z_dim = z_dim
		self.n_units = n_units
		layers = [nn.Linear(z_dim, n_units[0]), nn.ELU()]
		for i in range(1,len(n_units)):
			layers.append(nn.Linear(n_units[i-1],n_units[i]))
			layers.append(nn.ELU())
		layers.append(nn.Linear(n_units[-1],x_dim))
		self.net = nn.Sequential(*layers)

	def forward(self,z):
		return self.net(z)

class RealNVPLayer(nn.Module):
	def __init__(self, z_dim):
		super().__init__()
		self.z_dim = z_dim
		self.d1 = np.random.choice(np.arange(1,z_dim))
		self.d2 = self.z_dim - self.d1
		self.scale = nn.Sequential(nn.Linear(self.d1,self.d2), nn.ELU(), nn.Linear(self.d2,self.d2))
		self.shift = nn.Sequential(nn.Linear(self.d1,self.d2), nn.ELU(), nn.Linear(self.d2,self.d2))

	def forward(self,z):
		# z1 = z.select(dim=-1, np.arange(0,self.d1))
		# z2 = z.select(dim=-1, np.arange(self.d1,self.z_dim))
		z1,z2 = z.split([self.d1,self.d2], dim=-1)
		zout1 = z1
		zout2 = z2 * self.scale(z1).exp() + self.shift(z1)
		zout = torch.cat((zout1,zout2), dim=-1)
		return zout

	def inverse(self,z):
		z1,z2 = z.split([self.d1,self.d2], dim=-1)
		zout1 = z1
		zout2 = (z2 - self.shift(z1))/self.scale(z1).exp()
		return torch.cat((zout1,zout2), dim=-1)

	def jacobian(self,z):
		z1,_ = z.split([self.d1,self.d2], dim=-1)
		return self.scale(z1).sum(dim=-1).exp()

class PermLayer(nn.Module):
	def __init__(self, z_dim):
		super().__init__()
		self.perm = np.random.permutation(z_dim)

	def forward(self,z):
		#### Assumes z is (batch, z_dim) !!
		zout = z[:,self.perm]
		return zout

	def inverse(self,z):
		zout = z.new(z.size())
		zout[:,self_perm] = z
		return zout

	def jacobian(self,z):
		return 1.0

class Flow(nn.Module):
	def __init__(self, z_dim, n_layers):
		super().__init__()
		self.z_dim = z_dim
		self.n_layers = n_layers
		self.layers = []
		for i in range(n_layers):
			self.layers.append(RealNVPLayer(z_dim))
			self.layers.append(PermLayer(z_dim))

	def forward(self,z):
		for l in self.layers:
			z = l.forward(z)
		return z

	def inverse(self,z):
		for l in self.layers.reverse():
			z = l.inverse(z)
		return z

	def jacobian(self,z):
		J = 1.0
		for l in self.layers:
			J *= l.jacobian(z)
		return J


class Discriminator(nn.Module):
	def __init__(self, z_dim, n_units):
		super().__init__()
		self.z_dim = z_dim
		self.n_units = n_units
		layers = [nn.Linear(z_dim, n_units[0]), nn.ELU()]
		for i in range(1,len(n_units)):
			layers.append(nn.Linear(n_units[i-1],n_units[i]))
			layers.append(nn.ELU())
		layers.append(nn.Linear(n_units[-1],1))
		self.net = nn.Sequential(*layers)

	def forward(self,z):
		return self.net(z)