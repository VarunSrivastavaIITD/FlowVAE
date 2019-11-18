import matplotlib.pyplot as plt
from matplotlib import rc
import torchvision as tv
from datetime import datetime as dt
import torch
from math import sqrt


rc("text", usetex=True)
rc("font", size=16)

try:
    plt.switch_backend("qt5agg")
except ImportError:
    plt.switch_backend("TkAgg")


def plot_images(img_tensor, img_title=None, nrows=4):

    fig, ax = plt.subplots(constrained_layout=True)
    img_tensor = img_tensor.to("cpu")
    ax.axis("off")
    if not img_title:
        img_title = f'sampled image @ {dt.now().strftime("%M_%d %H:%M:%S")}'
    fig.suptitle(img_title)

    if img_tensor.ndim == 3:
        img_tensor.unsqueeze_(1)
    npimg = tv.utils.make_grid(img_tensor, nrow=nrows).numpy().transpose(1, 2, 0)

    ax.imshow(npimg)
    fig.set_tight_layout(True)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    return fig, ax


def log_ae_tensorboard_images(
    model, dataloader, writer, epoch, tag, shape=(28, 28), nrows=None, dataformat="NHW"
):
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            if nrows is None:
                nrows = round(sqrt(x.size(0)))
            xcap = model.predict(x).to("cpu").view(-1, *shape)
            writer.add_image(
                tag,
                tv.utils.make_grid(xcap.unsqueeze_(1), nrow=nrows).numpy(),
                global_step=epoch,
                dataformats="CHW",
            )
            # writer.add_images(tag, xcap, global_step=epoch, dataformats=dataformat)
    writer.flush()


def log_flow_tensorboard_images(
    flow_model,
    ae_model,
    # dataloader,
    writer,
    epoch,
    tag,
    shape=(28, 28),
    nsamples=100,
    dataformat="NHW",
):
    flow_model.eval()
    ae_model.eval()

    nrows = round(sqrt(nsamples))

    with torch.no_grad():
        # for (x,) in dataloader:

        z = flow_model.sample(nsamples)
        xcap = ae_model.decoder.predict(z).to("cpu").view(-1, *shape)
        writer.add_images(
            tag,
            tv.utils.make_grid(xcap.unsqueeze_(1), nrow=nrows).numpy(),
            global_step=epoch,
            dataformats="CHW",
        )
        # writer.add_images(tag, xcap, global_step=epoch, dataformats=dataformat)
    writer.flush()

