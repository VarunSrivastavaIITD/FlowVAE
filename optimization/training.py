import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from tqdm import tqdm
from appmetrics import metrics
from math import ceil
from utils.plotting import plot_images


def train_ae(epoch, dataloader, model, optimizer, writer, loss_func, **kwargs):
    # Initialization of model states, variables etc
    model.train()
    loss_meter = metrics.new_histogram(f"train_vae_loss_{epoch}")
    device = kwargs.get("device", next(model.parameters()).device)

    total_iters = (
        ceil(len(dataloader.dataset) / dataloader.batch_size)
        if not dataloader.drop_last
        else len(dataloader.dataset) // dataloader.batch_size
    )

    with tqdm(total=total_iters) as pbar:
        for batch_idx, (x, _) in enumerate(dataloader):

            optimizer.zero_grad()
            batch_size = x.size(0)
            x = x.to(device)

            xcap = model(x)
            loss = loss_func(xcap, x) / batch_size
            loss_meter.notify(loss.item())
            loss.backward()
            optimizer.step()

            pbar.set_postfix(avg_ae_loss=f'{loss_meter.get()["arithmetic_mean"]:.3e}')
            pbar.update()

    writer.add_scalar("Loss/AE/train/mean", loss_meter.get()["arithmetic_mean"], epoch)
    writer.add_scalar(
        "Loss/AE/train/std_dev", loss_meter.get()["standard_deviation"], epoch
    )


def evaluate_ae(epoch, dataloader, model, writer, loss_func, **kwargs):
    # Initialization of model states, variables etc
    model.eval()
    loss_meter = metrics.new_histogram(f"test_vae_loss_{epoch}")
    device = kwargs.get("device", next(model.parameters()).device)

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):

            batch_size = x.size(0)
            x = x.to(device)

            xcap = model(x)
            loss = loss_func(xcap, x) / batch_size
            loss_meter.notify(loss.item())

    writer.add_scalar("Loss/AE/test/mean", loss_meter.get()["arithmetic_mean"], epoch)
    writer.add_scalar(
        "Loss/AE/test/std_dev", loss_meter.get()["standard_deviation"], epoch
    )


def train_flow(epoch, dataloader, flow_model, ae_model, optimizer, writer, **kwargs):
    flow_model.train()
    loss_meter = metrics.new_histogram(f"train_flow_loss_{epoch}")
    flatten = kwargs.get("flatten", False)

    total_iters = (
        ceil(len(dataloader.dataset) / dataloader.batch_size)
        if not dataloader.drop_last
        else len(dataloader.dataset) // dataloader.batch_size
    )
    ae_model.eval()
    device = kwargs.get("device", next(ae_model.parameters()).device)

    with tqdm(total=total_iters) as pbar:
        for batch_idx, (x, y) in enumerate(dataloader):

            optimizer.zero_grad()
            batch_size = x.size(0)

            with torch.no_grad():
                x = ae_model.encoder(x.to(device))

            if flatten:
                x = x.view(batch_size, -1)

            z, prior_logprob, log_det = flow_model(x)
            logprob = prior_logprob + log_det
            loss = -torch.mean(prior_logprob + log_det)
            loss_meter.notify(loss.item())
            loss.backward()
            optimizer.step()

            pbar.set_postfix(avg_flow_loss=f'{loss_meter.get()["arithmetic_mean"]:.3e}')
            pbar.update()

    writer.add_scalar(
        "Loss/Flow/train/mean", loss_meter.get()["arithmetic_mean"], epoch
    )
    writer.add_scalar(
        "Loss/Flow/train/std_dev", loss_meter.get()["standard_deviation"], epoch
    )
