# Copyright (c) 2018 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
from models import vae_utils as ut
from models import VAE
from vae_train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import skimage.io

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',             default="mnist", help="Which dataset?")
parser.add_argument('--sup_mode',             default="none", help="Supervised or not")
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
args = parser.parse_args()
layout = [
    ('model={:s}',  'vae'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
train_loader, labeled_subset, _ = ut.get_mnist_data(args.dataset, device, use_test_subset=True)
vae = VAE(z_dim=args.z, name=model_name, sup=(args.sup_mode=="fullsup")).to(device)

if args.train:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train(model=vae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save,
          y_status=args.sup_mode)
    ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=args.train == 2)

else:
    ut.load_model_by_name(vae, global_step=args.iter_max, device=device)
    # ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=True)
#     images = vae.sample_x(100)
#     images_tiled = np.reshape(np.transpose(np.reshape(images.cpu().detach(), (10,10,28,28)), (0,2,1,3)), (280,280))
#     plt.imsave("images-vae.png", images_tiled, cmap="gray")
    with torch.no_grad():
        images = vae.sample_x(10000)
        images = images.cpu().detach().numpy()*255
        images = np.reshape(images,(-1,28,28)).astype(np.uint8)
        for i in range(10000):
            skimage.io.imsave(f'images/vae-samples/{i}.png', images[i])
