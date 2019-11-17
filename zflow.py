import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import random
from natsort import natsorted
from pathlib import Path
import os
import sys
import json
from datetime import datetime as dt
import normalizingflow.nf.flows as flows
from normalizingflow.nf.models import NormalizingFlowModel
from models.vaemodels import AutoEncoder
from utils.load_data import load_dataset
from optimization.training import train_ae, train_flow
from utils.plotting import log_ae_tensorboard_images, log_flow_tensorboard_images
import shutil
from utils.load_model import save_checkpoint
from tensorboardX import SummaryWriter
from tqdm import trange


def parse():
    gpu_available = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description="FlowVAE for latent space inference")

    parser.add_argument(
        "-z", "--zdim", type=int, default=10, help="Number of latent dimensions"
    )
    parser.add_argument(
        "-ve",
        "--vae_epochs",
        type=int,
        default=5,
        metavar="VAE_EPOCHS",
        help="number of epochs to train for vae model",
    )
    parser.add_argument(
        "-fe",
        "--flow_epochs",
        type=int,
        default=20,
        metavar="FLOW_EPOCHS",
        help="number of epochs to train for flow model",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=-1,
        metavar="EPOCHS",
        help="number of epochs to train, max of vae_epochs and flow_epochs by default",
    )
    parser.add_argument(
        "-f",
        "--flow",
        type=str,
        default="NSF_AR",
        choices=["Planar", "Radial", "RealNVP", "MAF", "NSF_AR", "NSF_CL"],
        help="Type of flows to use",
    )
    parser.add_argument(
        "-nf",
        "--num_flows",
        type=int,
        default=4,
        metavar="NUM_FLOWS",
        help="Number of flow layers, ignored in absence of flows",
    )
    parser.add_argument(
        "-s", "--save_iter", type=int, default=2, help="Save model every n iterations"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist"],
        metavar="DATASET",
        help="dataset choice.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="fraction of the training set to use for validation",
    )
    parser.add_argument(
        "--manual_seed",
        type=int,
        default=42,
        help="manual seed, if not given resorts to the dog number.",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=100,
        metavar="BATCH_SIZE",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.0005,
        metavar="LEARNING_RATE",
        help="learning rate",
    )
    parser.add_argument(
        "-nc", "--no_cuda", action="store_true", help="Use CPU for training/evaluation"
    )
    parser.add_argument("--pin_memory", action="store_true", help="Pin Memory in GPU")
    parser.add_argument(
        "-od",
        "--out_dir",
        type=str,
        default="results",
        metavar="OUT_DIR",
        help="output directory for model snapshots etc.",
    )
    parser.add_argument(
        "--writer_comment", default="", help="comment for tensorboard summary writer"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from earlier checkpoint"
    )
    parser.add_argument(
        "--initialize", action="store_true", help="initialize model with another model"
    )
    parser.add_argument("--model_name", help="Model Name for save files")
    parser.add_argument(
        "--model_path",
        # required=True if "--resume" or "--initialize" in sys.argv else False,
        help="save path used for resumes/reinitialization",
    )
    parser.add_argument(
        "--units",
        nargs="+",
        default=list(map(int, "300 300".split())),
        type=int,
        help="Hidden layer sizes for the autoencoder",
    )

    args = parser.parse_args()
    if args.no_cuda or not gpu_available:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda")
    if args.resume or args.initialize and not args.model_path:
        raise Exception("model path is required when resume or initialize is specified")

    if not args.model_name:
        existing_models = natsorted(
            [
                x.as_posix()
                for x in Path(args.out_dir).glob(f"*{args.flow}*")
                if x.is_dir()
            ]
        )
        if not existing_models:
            args.model_name = "_".join([args.flow, "1"])
        else:
            last_model = existing_models[-1]
            name_last_elem = last_model.split("_")[-1]

            if str.isdigit(name_last_elem):
                args.model_name = "_".join([args.flow, str(int(name_last_elem) + 1)])
            else:
                args.model_name = "_".join(
                    [args.flow, dt.now().strftime("%M_%D_%H_%M_%S")]
                )

    return args


def main():
    args = parse()
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
        while c != "y" or c != "n":
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

    writer = SummaryWriter(summary_writer_dir, comment=args.writer_comment)

    if hasattr(flows, args.flow):
        flow_model_template = getattr(flows, args.flow)

    flow_list = [flow_model_template(args.zdim) for _ in range(args.num_flows)]
    prior = torch.distributions.MultivariateNormal(
        torch.zeros(args.zdim), torch.eye(args.zdim)
    )

    train_loader, val_loader, test_loader, args = load_dataset(args, flatten=True)
    sample_dataset = torch.utils.data.TensorDataset(
        prior.sample((len(val_loader.dataset),))
    )
    sample_loader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        pin_memory=False,
    )

    flow_model = NormalizingFlowModel(prior, flow_list)
    ae_model = AutoEncoder(args.xdim, args.zdim, args.units, "binary")

    ae_optimizer = optim.Adam(ae_model.parameters(), args.learning_rate)
    flow_optimizer = optim.Adam(flow_model.parameters(), args.learning_rate)

    ae_loss = nn.BCEWithLogitsLoss(reduction="sum")

    total_epochs = np.max([args.vae_epochs, args.flow_epochs, args.epochs])

    if args.resume:
        raise NotImplementedError
    if args.initialize:
        raise NotImplementedError

    for epoch in trange(1, total_epochs + 1):
        if epoch <= args.vae_epochs:
            train_ae(epoch, train_loader, ae_model, ae_optimizer, writer, ae_loss)
            log_ae_tensorboard_images(
                ae_model, val_loader, writer, epoch, "AE/val/Images"
            )

        if epoch <= args.flow_epochs:
            train_flow(
                epoch, train_loader, flow_model, ae_model, flow_optimizer, writer
            )

            log_flow_tensorboard_images(
                flow_model,
                ae_model,
                sample_loader,
                writer,
                epoch,
                "Flow/sampled/Images",
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
