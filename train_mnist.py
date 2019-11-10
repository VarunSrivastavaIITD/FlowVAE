import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device ='cpu'


def get_mnist_data(device):
    preprocess = transforms.ToTensor()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("mnist_data", train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("mnist_data", train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True,
    )

    return train_loader, test_loader


train_loader, test_loader = get_mnist_data(device)
num_epochs = 100
x_dim = 784
z_dim = 20

encoder = models.Encoder(x_dim, z_dim, [300, 300]).to(device)
decoder = models.Decoder(z_dim, x_dim, [300, 300]).to(device)
flow = models.Flow(z_dim, 3).to(device)
discriminator = models.Discriminator(z_dim, [10, 10]).to(device)
g_optimizer = optim.Adam(flow.parameters(), lr=1e-3)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
ed_optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
)

for e in range(1,num_epochs+1):
    for batch in train_loader:
        encoder.train()
        decoder.train()
        flow.train()
        discriminator.train()
        x_batch, y_batch = batch
        batch_size = x_batch.size()[0]
        x_batch = x_batch.to(device).reshape(-1, 784).float() / 255
        labels = y_batch.new(np.eye(10)[y_batch.cpu()]).to(device).float()

        ## Train discriminator on real z's and fake z's
        ## min -log(G(real)) - log(1-G(fake))
        d_optimizer.zero_grad()
        real_z = encoder(x_batch).detach()
        real_loss = torch.nn.BCEWithLogitsLoss()(
            input=discriminator(real_z), target=torch.ones(batch_size, 1).to(device)
        ).sum()
        real_loss.backward()
        fake_z = flow(torch.rand(batch_size, z_dim).to(device)).detach()
        fake_loss = torch.nn.BCEWithLogitsLoss()(
            input=discriminator(fake_z), target=torch.zeros(batch_size, 1).to(device)
        ).sum()
        fake_loss.backward()
        d_optimizer.step()

        ## Train generator
        ## min -log(G(fake))
        g_optimizer.zero_grad()
        fake_z = flow(torch.rand(batch_size, z_dim).to(device))
        gen_loss = torch.nn.BCEWithLogitsLoss()(
            input=discriminator(real_z), target=torch.ones(batch_size, 1).to(device)
        ).sum()
        gen_loss.backward()
        g_optimizer.step()

        ## Train encoder-decoder
        ## min -E_{q(z|x)} log(p(x|z))
        ed_optimizer.zero_grad()
        z = encoder(x_batch)
        ed_loss = torch.nn.BCEWithLogitsLoss()(input=decoder(z), target=x_batch).sum()
        ed_loss.backward()
        ed_optimizer.step()

        # pbar.set_postfix(loss='{:.2e}'.format(loss))
    print(
        "Epoch {} : Disc loss = {:.2e} Gen loss = {:.2e} E-D loss = {:.2e}".format(
            e, real_loss + fake_loss, gen_loss, ed_loss
        )
    )

    flow.eval()
    decoder.eval()
    with torch.no_grad():
        z = flow(torch.rand(200, z_dim).to(device))
        images = torch.bernoulli(torch.sigmoid(decoder(z)))
    images_tiled = np.reshape(np.transpose(np.reshape(images.cpu().detach(), (10,20,28,28)), (0,2,1,3)), (280,560))
    plt.imsave('images/{}.png'.format(e), images_tiled, cmap='gray')

