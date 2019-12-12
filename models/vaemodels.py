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
    def __init__(self, z0_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential( #64,1,1
            Reshape(z0_dim,1,1),
            nn.ConvTranspose2d(z0_dim, 48, kernel_size=2, stride=1, padding=0), #48,2,2
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1), # 32,4,4
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1), # 24,8,8
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1), #16,16,16
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, z_dim[0], kernel_size=4, stride=2, padding=z_dim[-1]-7) #z_dim[0],8,8
        )
        self.z_dim = z_dim

    def forward(self, z):
        return self.net(z)#.view(-1, np.prod(self.z_dim))


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
    def __init__(self, x_dim, z_dim, n_units, output_dist="binary", sup=False):
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
        if sup: self.encoder = Encoder(x_dim+10, z_dim, n_units)
        else: self.encoder = Encoder(x_dim, z_dim, n_units)
        if output_dist.lower() == "binary":
            if sup:
                self.decoder = Decoder(z_dim+10, x_dim, n_units, "binary")
            else: self.decoder = Decoder(z_dim, x_dim, n_units, "binary")

    def forward(self, x):
        z = self.encoder(x)
        xcap = self.decoder(z)
        return xcap

    def predict(self, x):
        x = self.encoder(x)
        x = self.decoder.predict(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class ConvAutoEncoder(nn.Module):
    def __init__(self, in_channels=3, image_size=(32,32), activation=nn.Sigmoid()):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=2, padding=1),  # b, 16, 16, 16
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1
            )  # b, 8, 8, 8
        )
        self.decoder = nn.Sequential(
#             nn.Linear(z_dim, 8*(image_size[0]//4)*(image_size[1]//4)),
#             nn.BatchNorm1d(8*(image_size[0]//4)*(image_size[1]//4)),
#             nn.ReLU(),
#             Reshape(8,(image_size[0]//4),(image_size[1]//4)),
            nn.ConvTranspose2d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # b, 16, 16, 16
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # b, 16, 32, 32
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=in_channels, kernel_size=1, stride=1),  # b, 3, 32, 32
        )
        if activation != None:
            self.decoder.add_module("activation", activation)
        self.decoder.predict = self.decoder.forward
        self.z_dim = (4, (image_size[0] // 4), (image_size[1] // 4))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        z = self.encoder(x)
        return z#.view(-1, np.prod(self.z_dim))

    def decode(self, z):
        x = self.decoder(z)#.view(-1, *self.z_dim))
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

class ConvDiscriminator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential( #8,8,8
            nn.Conv2d(z_dim[0], z_dim[0]*2, kernel_size=z_dim[-1]-4, stride=2, padding=1), #16,4,4
            nn.ELU(),
            nn.Conv2d(z_dim[0]*2, z_dim[0]*4, kernel_size=3, stride=1, padding=0), #32,4,4
            nn.ELU(),
            nn.Conv2d(z_dim[0]*4, z_dim[0]*8, kernel_size=4, stride=2, padding=1), #32,2,2
            nn.ELU(),
            Reshape(32),
            nn.Linear(32,1)
        )
    def forward(self, z):
        return self.net(z)

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2, sup=False):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
#         nn = getattr(nns, nn)
        self.enc = vae_nn.Encoder(self.z_dim, 10*int(sup))
        self.dec = vae_nn.Decoder(self.z_dim, 10*int(sup))

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, y=None):
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
        m,v = self.enc.encode(x,y)
        z_batch = ut.sample_gaussian(m,v)
        rec = - torch.mean(ut.log_bernoulli_with_logits(x, logits=self.dec.decode(z_batch,y)))
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

    def loss(self, x, y=None):
        nelbo, kl, rec = self.negative_elbo_bound(x,y)
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

    def compute_sigmoid_given(self, z, y=None):
        logits = self.dec.decode(z,y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_sup(self, batch=10):
        z = self.sample_z(10).repeat(10,1)
        y = torch.eye(10)[torch.arange(10)].repeat_interleave(10,0)
        return self.sample_x_given(z,y)
        
    def sample_x_given(self, z, y=None):
        return torch.bernoulli(self.compute_sigmoid_given(z,y))