import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
from torch.nn import functional as F
from . import vae_utils as ut
from . import vae_nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

class Encoder(nn.Module):
    """docstring for Encoder"""

    def __init__(self, x_dim, z_dim, n_units):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_units = n_units
        layers = [nn.Linear(x_dim, n_units[0]), nn.BatchNorm1d(n_units[0]), nn.ELU()]
        for i in range(1, len(n_units)):
            layers.append(nn.Linear(n_units[i - 1], n_units[i]))
            layers.append(nn.BatchNorm1d(n_units[i - 1]))            
            layers.append(nn.ELU())
        layers.append(nn.Linear(n_units[-1], z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvGenerator(nn.Module):
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim,z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ELU(),
            nn.Linear(z_dim, np.prod(x_dim)),
            nn.BatchNorm1d(np.prod(x_dim)),
            nn.ELU(),
            Reshape(*x_dim),
            nn.Conv2d(x_dim[0], x_dim[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(x_dim[0]),
            nn.ELU(),
            nn.Conv2d(x_dim[0], x_dim[0], kernel_size=3, padding=1)
        )
        self.x_dim = x_dim

    def forward(self,z):
        return self.net(z).view(-1,np.prod(self.x_dim))

class Decoder(nn.Module):
    def __init__(self, z_dim, x_dim, n_units, x_type="binary"):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.n_units = n_units
        layers = [nn.Linear(z_dim, n_units[0]), nn.BatchNorm1d(n_units[0]), nn.ELU()]
        for i in range(1, len(n_units)):
            layers.append(nn.Linear(n_units[i - 1], n_units[i]))
            layers.append(nn.BatchNorm1d(n_units[i - 1]))
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
        x_dim = np.squeeze(x_dim)
        z_dim = np.squeeze(z_dim)
        if x_dim.size > 1:
            raise Exception("Multidimensional input where scalar was expected")
        x_dim = x_dim.item()

        if z_dim.size > 1:
            raise Exception("Multidimensional input where scalar was expected")
        z_dim = z_dim.item()
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
    def __init__(self, in_channels=3, image_size=(32,32), z_dim=20, activation=nn.Sigmoid()):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=1),  # b, 16, 16, 16
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1
            ),  # b, 8, 8, 8
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            Reshape(8*(image_size[0]//4)*(image_size[1]//4)),
            nn.Linear(8*(image_size[0]//4)*(image_size[1]//4), z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 8*(image_size[0]//4)*(image_size[1]//4)),
            nn.BatchNorm1d(8*(image_size[0]//4)*(image_size[1]//4)),
            nn.ReLU(),
            Reshape(8,(image_size[0]//4),(image_size[1]//4)),
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # b, 16, 16, 16
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # b, 16, 32, 32
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=in_channels, kernel_size=1, stride=1),  # b, 3, 32, 32
        )
        if activation!=None:
            self.decoder.add_module("activation", activation)
        self.z_dim = z_dim#(8,(image_size[0]//4),(image_size[1]//4))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        z = self.encoder(x)
        return z#.view(-1,np.prod(self.z_dim))

    def decode(self,z):
        x = self.decoder(z)#.view(-1,*self.z_dim))
        return x

    def predict(self, x):
        return self.forward(x)


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

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
#         nn = getattr(nns, nn)
        self.enc = vae_nn.Encoder(self.z_dim)
        self.dec = vae_nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        m,v = self.enc.encode(x)
        z_batch = ut.sample_gaussian(m,v)
        rec = - torch.mean(ut.log_bernoulli_with_logits(x, logits=self.dec.decode(z_batch)))
        kl = torch.mean(ut.kl_normal(m,v,torch.zeros_like(m),torch.ones_like(v)))
        nelbo = kl + rec
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        m,v = self.enc.encode(x) #(batch,dim_z)
        x_rep = x.unsqueeze(dim=1).expand(-1,iw,-1) #(batch,iw,dim_x)
        m_rep = m.unsqueeze(dim=1).expand(-1,iw,-1) #(batch,iw,dim_z)
        v_rep = v.unsqueeze(dim=1).expand(-1,iw,-1) #(batch,iw,dim_z)
        z_batch = ut.sample_gaussian(m_rep,v_rep) #(batch,iw,dim_z)
        kl = torch.mean(ut.kl_normal(m,v,torch.zeros_like(m),torch.ones_like(v)))
        rec = - torch.mean(ut.log_bernoulli_with_logits(x_rep, logits=self.dec.decode(z_batch)))
        niwae = - torch.mean(ut.log_mean_exp(
        ut.log_normal(z_batch,torch.zeros_like(z_batch),torch.ones_like(z_batch)) #log p_{theta}(z)
        + ut.log_bernoulli_with_logits(x_rep, logits=self.dec.decode(z_batch)) # log p_{theta}(x|z)
        - ut.log_normal(z_batch,m_rep,v_rep), dim=-1) ) #log q_{phi}(z|x)
        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))