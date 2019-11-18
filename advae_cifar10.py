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

def get_cifar10_data(device):
    def my_transform(x):
        return x.to(device)
    preprocess = transforms.Compose([transforms.ToTensor(),my_transform])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("data", train=True, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("data", train=False, download=True, transform=preprocess),
        batch_size=100,
        shuffle=True,
    )

    return train_loader, test_loader

train_loader, test_loader = get_cifar10_data(device)
num_epochs1 = 50
num_epochs2 = 100
save_path = 'checkpoints/'

ae = models.ConvAutoEncoder().to(device)
ae_optimizer = optim.Adam(ae.parameters(), lr=1e-3)

# writer = SummaryWriter()
# First train encoder and decoder
print("Training Encoder-Decoder .............")
for e in range(1, num_epochs1 + 1):
    for batch in train_loader:
        ae.train()
        x_batch, y_batch = batch
        batch_size = x_batch.size()[0]
        labels = torch.eye(10)[y_batch.cpu()].to(device).float()

        ## Train encoder-decoder
        ## min -E_{q(z|x)} log(p(x|z))
        ae_optimizer.zero_grad()
        x_out = ae(x_batch)
        ae_loss = torch.nn.MSELoss(reduction='none')(input=x_out, target=x_batch).sum(-1).mean()
        ae_loss.backward()
        ae_optimizer.step()

    with torch.no_grad():
        x_batch = next(iter(test_loader))[0]
        ae.eval()
        x_out = ae(x_batch)
        test_loss = torch.nn.MSELoss(reduction='none')(input=x_out, target=x_batch).sum(-1).mean()
    images = x_out.cpu().detach().numpy()
    images_tiled = np.reshape(np.transpose(np.reshape(images, (10,10,3,32,32)), (0,3,1,4,2)), (320,320,3))
    plt.imsave("images-conv-ae/{}.png".format(e), images_tiled)
    print(
        "Epoch {} : E-D train loss = {:.2e} test loss = {:.2e}".format(
            e, ae_loss, test_loss
        )
    )
    # writer.add_scalars('losses', {'train':ed_loss, 'test':test_loss}, e)
    if e%5==0:
        checkpoint_dict = {'epoch':e, 'model':ae.state_dict(), 'optimizer':ae_optimizer.state_dict()}
        fname = f'conv-ae_{e}'
        save_checkpoint(checkpoint_dict, save_path, fname)