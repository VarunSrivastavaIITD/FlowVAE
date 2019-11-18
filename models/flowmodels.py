import numpy as np
import torch
import torch.nn.functional as F
from torch import autograd, nn, optim
from torch.nn import functional as F
from sylvester_flow.models import flows


class RealNVPLayer(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.d1 = np.random.choice(np.arange(1, z_dim))
        self.d2 = self.z_dim - self.d1
        self.scale = nn.Sequential(
            nn.Linear(self.d1, self.d2), nn.ELU(), nn.Linear(self.d2, self.d2)
        )
        self.shift = nn.Sequential(
            nn.Linear(self.d1, self.d2), nn.ELU(), nn.Linear(self.d2, self.d2)
        )

    def forward(self, z):
        # z1 = z.select(dim=-1, np.arange(0,self.d1))
        # z2 = z.select(dim=-1, np.arange(self.d1,self.z_dim))
        z1, z2 = z.split([self.d1, self.d2], dim=-1)
        zout1 = z1
        zout2 = z2 * self.scale(z1).exp() + self.shift(z1)
        zout = torch.cat((zout1, zout2), dim=-1)
        return zout

    def inverse(self, z):
        z1, z2 = z.split([self.d1, self.d2], dim=-1)
        zout1 = z1
        zout2 = (z2 - self.shift(z1)) / self.scale(z1).exp()
        return torch.cat((zout1, zout2), dim=-1)

    def jacobian(self, z):
        z1, _ = z.split([self.d1, self.d2], dim=-1)
        return self.scale(z1).sum(dim=-1).exp()


class PermLayer(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.perm = nn.Parameter(
            torch.from_numpy(np.random.permutation(z_dim)), requires_grad=False
        )

    def forward(self, z):
        #### Assumes z is (batch, z_dim) !!
        zout = z[:, self.perm]
        return zout

    def inverse(self, z):
        zout = z.new(z.size())
        zout[:, self_perm] = z
        return zout

    def jacobian(self, z):
        return 1.0


class Flow(nn.Module):
    def __init__(self, z_dim, n_layers):
        super().__init__()
        self.z_dim = z_dim
        self.n_layers = n_layers
        layers = []
        for i in range(n_layers):
            layers.append(RealNVPLayer(z_dim))
            layers.append(PermLayer(z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

    def inverse(self, z):
        for l in net:
            z = l.inverse(z)
        return z

    def jacobian(self, z):
        J = 1.0
        for l in net:
            J *= l.jacobian(z)
        return J


class VPlanarFlow(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.log_det_j = 0.0

        # Flow parameters
        flow = flows.Planar
        self.num_flows = args.num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        self.amor_w = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module("flow_" + str(k), flow_k)

    def encode(self, h):
        """
        Encoder that ouputs parameters for base distribution of flow parameters.
        """

        batch_size = h.size(0)

        # return amortized u an w for all flows
        u = self.amor_u(h).view(batch_size, self.num_flows, self.z_size, 1)
        w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.z_size)
        b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)

        return u, w, b

    def forward(self, x):
        self.log_det_j = 0.0

        u, w, b = self.encode(x)

        # Sample z_0
        z = [x]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, "flow_" + str(k))
            z_k, log_det_jacobian = flow_k(
                z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :]
            )
            z.append(z_k)
            self.log_det_j += log_det_jacobian

        return self.log_det_j, z[0], z[-1]
