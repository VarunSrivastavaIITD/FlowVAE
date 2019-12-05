import torch
import argparse
from natsort import natsorted
from datetime import datetime as dt
from pathlib import Path


def parse():
    gpu_available = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description="FlowVAE for latent space inference")

    parser.add_argument(
        "-z", "--zdim", type=int, default=10, help="Number of latent dimensions"
    )
    parser.add_argument(
        "-ve",
        "--vae-epochs",
        type=int,
        default=5,
        metavar="VAE_EPOCHS",
        help="number of epochs to train for vae model",
    )
    parser.add_argument(
        "-fe",
        "--flow-epochs",
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
        "--num-flows",
        type=int,
        default=4,
        metavar="NUM_FLOWS",
        help="Number of flow layers, ignored in absence of flows",
    )
    parser.add_argument(
        "-s", "--save-iter", type=int, default=2, help="Save model every n iterations"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        metavar="DATASET",
        help="dataset choice.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="fraction of the training set to use for validation",
    )
    parser.add_argument(
        "--manual-seed",
        type=int,
        default=42,
        help="manual seed, if not given resorts to the dog number.",
    )

    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=100,
        metavar="BATCH_SIZE",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.0005,
        metavar="LEARNING_RATE",
        help="learning rate",
    )
    parser.add_argument(
        "-nc", "--no-cuda", action="store_true", help="Use CPU for training/evaluation"
    )
    parser.add_argument("--pin_memory", action="store_true", help="Pin Memory in GPU")
    parser.add_argument(
        "-od",
        "--out-dir",
        type=str,
        default="results",
        metavar="OUT_DIR",
        help="output directory for model snapshots etc.",
    )
    parser.add_argument(
        "--writer-comment", default="", help="comment for tensorboard summary writer"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training from earlier checkpoint"
    )
    parser.add_argument(
        "--initialize", action="store_true", help="initialize model with another model"
    )
    parser.add_argument("--model_name", help="Model Name for save files")
    parser.add_argument(
        "--model-path",
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
    parser.add_argument(
        "--flatten", action="store_true", help="flatten the input image/domain"
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
