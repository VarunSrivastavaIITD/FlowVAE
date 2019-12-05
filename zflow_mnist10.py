import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from natsort import natsorted
from tensorboardX import SummaryWriter
from tqdm import trange
from skimage.io import imsave

import normalizingflow.nf.flows as flows
from models import AutoEncoder, ConvAutoEncoder
from normalizingflow.nf.models import NormalizingFlowModel
from optimization import evaluate_ae, train_ae, train_flow
from utils import (
    load_dataset,
    save_checkpoint,
    parse,
    log_ae_tensorboard_images,
    log_flow_tensorboard_images,
)


def main():
    args = parse()

    # set random seeds
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    # prepare output directories
    base_dir = Path(args.out_dir)
    model_dir = base_dir.joinpath(args.model_name)
    if (args.resume or args.initialize) and not model_dir.exists():
        raise Exception("Model directory for resume does not exist")
    if not (args.resume or args.initialize) and model_dir.exists():
        c = ""
        while c != "y" and c != "n":
            c = input("Model directory already exists, overwrite?").strip()

        if c == "y":
            shutil.rmtree(model_dir)
        else:
            sys.exit(0)
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_writer_dir = model_dir.joinpath("runs")
    summary_writer_dir.mkdir(exist_ok=True)
    save_path = model_dir.joinpath("checkpoints")
    save_path.mkdir(exist_ok=True)

    # prepare summary writer
    writer = SummaryWriter(summary_writer_dir, comment=args.writer_comment)

    # prepare data
    train_loader, val_loader, test_loader, args = load_dataset(
        args, flatten=args.flatten
    )

    # prepare flow model
    if hasattr(flows, args.flow):
        flow_model_template = getattr(flows, args.flow)

    flow_list = [flow_model_template(args.zdim) for _ in range(args.num_flows)]
    prior = torch.distributions.MultivariateNormal(
        torch.zeros(args.zdim), torch.eye(args.zdim)
    )
    flow_model = NormalizingFlowModel(prior, flow_list).to(args.device)

    # prepare autoencoder
    if args.dataset == "mnist":
        ae_model = AutoEncoder(args.xdim, args.zdim, args.units, "binary").to(
            args.device
        )
    elif args.dataset == "cifar10":
        ae_model = ConvAutoEncoder().to(args.device)

    # setup optimizers
    ae_optimizer = optim.Adam(ae_model.parameters(), args.learning_rate)
    flow_optimizer = optim.Adam(flow_model.parameters(), args.learning_rate)

    # setup loss
    if args.dataset == "mnist":
        args.imshape = (1, 28, 28)
        args.zshape = (args.zdim,)
        ae_loss = nn.BCEWithLogitsLoss(reduction="sum").to(args.device)
    elif args.dataset == "cifar10":
        args.imshape = (3, 32, 32)
        args.zshape = (8, 8, 8)
        ae_loss = nn.MSELoss(reduction="sum").to(args.device)

    total_epochs = np.max([args.vae_epochs, args.flow_epochs, args.epochs])

    if args.resume:
        raise NotImplementedError
    if args.initialize:
        raise NotImplementedError

    # training loop
    for epoch in trange(1, total_epochs + 1):
        if epoch <= args.vae_epochs:
            train_ae(
                epoch,
                train_loader,
                ae_model,
                ae_optimizer,
                writer,
                ae_loss,
                device=args.device,
            )
            log_ae_tensorboard_images(
                ae_model,
                val_loader,
                writer,
                epoch,
                "AE/val/Images",
                xshape=args.imshape,
            )
            # evaluate_ae(epoch, test_loader, ae_model, writer, ae_loss)

        if epoch <= args.flow_epochs:
            train_flow(
                epoch,
                train_loader,
                flow_model,
                ae_model,
                flow_optimizer,
                writer,
                device=args.device,
            )

            if epoch == args.flow_epochs:
                p = Path(f"images/mnist/{args.model_name}")
                p.mkdir(parents=True, exist_ok=True)
                n_samples = 10000

                print("final epoch images")
                flow_model.eval()
                ae_model.eval()
                with torch.no_grad():
                    z = flow_model.sample(n_samples)
                    z = z.to(next(ae_model.parameters()).device)
                    xcap = (
                        ae_model.decoder.predict(z)
                        .to("cpu")
                        .view(-1, *args.imshape)
                        .numpy()
                    )
                xcap = (np.rint(xcap) * int(255)).astype(np.uint8)
                for i, im in enumerate(xcap):
                    imsave(f'{p.joinpath(f"im_{i}.png").as_posix()}', np.squeeze(im))

                sys.exit(0)

            log_flow_tensorboard_images(
                flow_model,
                ae_model,
                writer,
                epoch,
                "Flow/sampled/Images",
                xshape=args.imshape,
                zshape=args.zshape,
            )

        if epoch % args.save_iter == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "ae_optimizer": ae_optimizer.state_dict(),
                "flow_optimizer": flow_optimizer.state_dict(),
                "ae_model": ae_model.state_dict(),
                "flow_model": flow_model.state_dict(),
            }
            fname = f"model_{epoch}.pt"
            save_checkpoint(checkpoint_dict, save_path, fname)

    writer.close()


if __name__ == "__main__":
    main()
