import torch
from utils import constants
from pathlib import Path
from datetime import datetime as dt


def save_checkpoint(
    checkpoint_dict, save_path, fname, append_datetime=False, strftime=None
):
    save_path = Path(save_path)
    fname = Path(fname)
    fbase, fext = fname.stem, fname.suffix
    if append_datetime:
        if not strftime:
            strftime = "%M_%D_%H_%M_%S"
        fbase = "_".join([fbase, dt.now().strftime(strftime)])
    fpath = Path(fbase + fext)
    fullpath = save_path.joinpath(fpath)
    torch.save(
        checkpoint_dict, fullpath.as_posix(), pickle_protocol=constants.PICKLE_PROTOCOL
    )


def load_checkpoint(save_path, fname, device="cpu"):
    save_path = Path(save_path)
    fname = Path(fname)
    fullpath = save_path.joinpath(fname)
    return torch.load(fullpath.as_posix(), map_location=device)

