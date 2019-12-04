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

import normalizingflow.nf.flows as flows
from models.vaemodels import AutoEncoder
from normalizingflow.nf.models import NormalizingFlowModel
from optimization.training import evaluate_ae, train_ae, train_flow
from utils.load_data import load_dataset
from utils.load_model import save_checkpoint
from utils.parse import parse
from utils.plotting import log_ae_tensorboard_images, log_flow_tensorboard_images


def main():
    pass


if __name__ == "__main__":
    main()
