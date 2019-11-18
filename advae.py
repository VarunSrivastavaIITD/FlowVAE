import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import models
from torch.utils.tensorboard import SummaryWriter
from utils.load_model import save_checkpoint, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mnist_data(device):
    def my_transform(x):
        return x.to(device).reshape(784)
    preprocess = transforms.Compose([transforms.ToTensor(),my_transform])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True,
    )

    return train_loader, test_loader


train_loader, test_loader = get_mnist_data(device)
num_epochs1 = 50
num_epochs2 = 100
x_dim = 784
z_dim = 10
save_path = 'checkpoints/'

encoder = models.Encoder(x_dim, z_dim, [300, 300]).to(device)
decoder = models.Decoder(z_dim, x_dim, [300, 300]).to(device)
generator = models.Encoder(z_dim, z_dim, [300,300]).to(device)
discriminator = models.Discriminator(z_dim, [20,20]).to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
ed_optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
)

writer = SummaryWriter()
# # First train encoder and decoder
# print("Training Encoder-Decoder .............")
# for e in range(1, num_epochs1 + 1):
#     for batch in train_loader:
#         encoder.train()
#         decoder.train()
#         x_batch, y_batch = batch
#         batch_size = x_batch.size()[0]
#         labels = torch.eye(10)[y_batch.cpu()].to(device).float()

#         ## Train encoder-decoder
#         ## min -E_{q(z|x)} log(p(x|z))
#         ed_optimizer.zero_grad()
#         z = encoder(x_batch)
#         x_out = decoder(z)
#         ed_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(input=x_out, target=x_batch).sum(-1).mean()
#         ed_loss.backward()
#         ed_optimizer.step()

#     with torch.no_grad():
#         # z = generator(torch.rand(100, z_dim).to(device))
#         x_batch = next(iter(test_loader))[0]
#         encoder.eval()
#         decoder.eval()
#         z = encoder(x_batch)
#         x_out = decoder(z)
#         images = torch.bernoulli(torch.sigmoid(x_out))
#         test_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(input=x_out, target=x_batch).sum(-1).mean()
#     images_tiled = np.reshape(
#         np.transpose(np.reshape(images.cpu().detach(), (10, 10, 28, 28)), (0, 2, 1, 3)),
#         (280, 280),
#     )
#     plt.imsave("images-ae/{}.png".format(e), images_tiled, cmap="gray")
#     print(
#         "Epoch {} : E-D train loss = {:.2e} test loss = {:.2e}".format(
#             e, ed_loss, test_loss
#         )
#     )
#     writer.add_scalars('losses', {'train':ed_loss, 'test':test_loss}, e)

#     if e%5==0:
#         checkpoint_dict = {
#             'epoch':e,
#             'encoder':encoder.state_dict(),
#             'decoder':decoder.state_dict(),
#             'ed_optimizer':ed_optimizer.state_dict()
#         }
#         fname = f'enc-dec_{e}'
#         save_checkpoint(checkpoint_dict, save_path, fname)
# writer.close()

fname = 'enc-dec_20'
enc_dec = load_checkpoint(save_path, fname, device)
encoder.load_state_dict(enc_dec['encoder'])
decoder.load_state_dict(enc_dec['decoder'])

print("Training Discriminator and generator")
for e in range(1, num_epochs2 + 1):
    ## Train discriminator on real z's and fake z's
    ## min -log(G(real)) - log(1-G(fake))
    for i in range(1):
        for batch in train_loader:
            x_batch, y_batch = batch
            batch_size = x_batch.size()[0]
            labels = torch.eye(10)[y_batch.cpu()].to(device).float()
            generator.train()
            discriminator.train()
            d_optimizer.zero_grad()
            real_z = encoder(x_batch).detach()
            real_labels = torch.rand(batch_size,1).to(device)*0.5 + 0.7
            real_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
                input=discriminator(real_z), target=real_labels
            ).sum(-1).mean()
            real_loss.backward()
            d_optimizer.step()
            d_optimizer.zero_grad()
            fake_z = generator(torch.randn(batch_size, z_dim).to(device)).detach()
            # fake_z = torch.randn(batch_size, z_dim).to(device)
            fake_labels = torch.rand(batch_size,1).to(device)*0.3
            fake_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
                input=discriminator(fake_z), target=fake_labels
            ).sum(-1).mean()
            disc_fake_output=discriminator(fake_z)
            fake_loss.backward()
            d_optimizer.step()
    ## Train generator
    ## min -log(G(fake))
    for i in range(2):
        g_optimizer.zero_grad()
        fake_z = generator(torch.randn(batch_size, z_dim).to(device))
        gen_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(
            input=discriminator(fake_z), target=torch.ones(batch_size, 1).to(device)
        ).sum(-1).mean()
        gen_loss.backward()
        g_optimizer.step()

    print(
        "Epoch {} : Real loss = {:.2e} Fake loss = {:.2e} Gen loss = {:.2e}".format(
            e, real_loss, fake_loss, gen_loss
        )
    )

    disc_grad_norm = 0
    for p in discriminator.parameters():
        param_norm = p.grad.data.norm(2)
        disc_grad_norm += param_norm.item() ** 2
    gen_grad_norm = 0
    for p in generator.parameters():
        param_norm = p.grad.data.norm(2)
        gen_grad_norm += param_norm.item() ** 2
    print(f'Disc grad norm {disc_grad_norm}')
    print(f'Gen grad norm {gen_grad_norm}')
    
    generator.eval()
    with torch.no_grad():
        z = generator(torch.randn(100, z_dim).to(device))
        # batch = next(iter(test_loader))[0]
        # z = encoder(batch)
        images = torch.bernoulli(torch.sigmoid(decoder(z)))
    images_tiled = np.reshape(
        np.transpose(np.reshape(images.cpu().detach(), (10, 10, 28, 28)), (0, 2, 1, 3)),
        (280, 280),
    )
    plt.imsave("images-gen/{}.png".format(e), images_tiled, cmap="gray")

    if e%10==0:
        checkpoint_dict = {
            'epoch':e,
            'generator':generator.state_dict(),
            'discriminator':discriminator.state_dict(),
            'g_optimizer':g_optimizer.state_dict(),
            'd_optimizer':d_optimizer.state_dict()
        }
        fname = f'gen-disc_{e}'
        save_checkpoint(checkpoint_dict, save_path, fname)