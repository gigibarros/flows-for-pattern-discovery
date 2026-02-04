import yaml
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from pattern_flows.data.toy_data import get_toy_dataset

def load_config(path: str) -> dict:
    """Load config file as a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_ckpt_file(config, model_name, epoch=None):
    """Load checkpoint file."""
    dir = Path(config["paths"]["output_dir"])
    dir.mkdir(exist_ok=True, parents=True)

    if epoch:
        return dir / f"{model_name}_epoch={epoch}.pth"
    else:
        return dir / f"{model_name}.pth"

def get_save_dir(output_dir):
    """Load save directory."""
    save_dir = Path(output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    return save_dir

def plot_losses(train_losses, valid_losses, save_dir, save_tag=""):
    """Plot training and validation losses."""
    plt.plot(train_losses, label="Training loss")
    plt.plot(valid_losses, label="Validation losss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, min(train_losses[0], 10))
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / f"losses_{save_tag}.png")
    plt.close()

def _plot_flow_samples(xs, y, vmin, vmax, save_dir, save_tag=""):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for idx, ax in enumerate(axs.flat):
        im = ax.imshow(xs[idx].T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Activity")
        ax.set_xlabel("Time")

    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.suptitle(f"Generative samples conditioned on y = {y}")

    fig.savefig(save_dir / f"samples_{save_tag}_y={y}.png")
    plt.close(fig)
    
def sample_flow(vae, tarflow, config, save_dir, save_tag=""):
    num_timesteps     = config["data"]["num_timesteps"]
    num_neurons       = config["data"]["num_neurons"]
    token_size        = config["tarflow"]["token_size"]
    z_dim             = config["tarflow"]["z_dim"]
    device            = config["device"]["type"]
    samples_per_y     = 4

    samples = torch.randn(samples_per_y, z_dim, token_size).to(device)

    x_0s = []
    x_1s = []

    for y in [0, 1]:
        with torch.no_grad():
            zs = tarflow.reverse(samples, y)  # shape : (samples_per_class, z_dim, token_size)
            
        for z in zs:
            z = z.squeeze(-1)  # shape : (z_dim,)

            with torch.no_grad():
                x = vae.decoder(z)  # shape : (input_dim,)

            x = x.view(num_timesteps, num_neurons)  # shape : (num_timesteps, num_neurons)

            if y == 0:
                x_0s.append(x.detach().cpu().numpy())
            if y == 1:
                x_1s.append(x.detach().cpu().numpy())

    vmin = min(x.min() for x in x_0s + x_1s)
    vmax = max(x.max() for x in x_0s + x_1s)

    _plot_flow_samples(x_0s, 0, vmin, vmax, save_dir, save_tag)
    _plot_flow_samples(x_1s, 1, vmin, vmax, save_dir, save_tag)

def _plot_vae_samples(x0_in, x0_hat, x1_in, x1_hat, save_dir, epoch=None):
    """Plot samples from dataset."""

    # set consistent color scale across all plotss
    vmin = min(x0_in.min(), x0_hat.min(), x1_in.min(), x1_hat.min())
    vmax = max(x0_in.max(), x0_hat.max(), x1_in.max(), x1_hat.max())

    f = plt.figure(figsize=(12, 8))

    ax = f.add_subplot(2, 2, 1)
    im = ax.imshow(x0_in.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_ylabel("neuron activity")
    ax.set_xlabel("time")
    ax.set_title("x input (y = 0)")
    plt.colorbar(im, ax=ax, label="activity")

    ax = f.add_subplot(2, 2, 2)
    im = ax.imshow(x0_hat.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_ylabel("neuron activity")
    ax.set_xlabel("time")
    ax.set_title("x reconstruction (y = 0)")
    plt.colorbar(im, ax=ax, label="activity")

    ax = f.add_subplot(2, 2, 3)
    im = ax.imshow(x1_in.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_ylabel("neuron activity")
    ax.set_xlabel("time")
    ax.set_title("x input (y = 1)")
    plt.colorbar(im, ax=ax, label="activity")

    ax = f.add_subplot(2, 2, 4)
    im = ax.imshow(x1_hat.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_ylabel("neuron activity")
    ax.set_xlabel("time")
    ax.set_title("x reconstruction (y = 1)")
    plt.colorbar(im, ax=ax, label="activity")

    plt.tight_layout()
    plt.savefig(save_dir / f"vae_samples_epoch={epoch}.png" if epoch is not None else save_dir / "vae_samples.png")
    plt.show()

    plt.close()

def sample_vae(config, vae_model, save_dir, epoch=None):
    num_timesteps  = config["data"]["num_timesteps"]
    num_neurons    = config["data"]["num_neurons"]

    dataset = get_toy_dataset(config)
    
    # generate one sample each for y = 0 and y = 1
    x0_in = dataset.generate_x(y=0)
    x1_in = dataset.generate_x(y=1)

    x0_flat = torch.flatten(x0_in)
    x1_flat = torch.flatten(x1_in)

    h0, _ = vae_model.encoder(x0_flat)
    h1, _ = vae_model.encoder(x1_flat)

    x0_hat = vae_model.decoder(h0)
    x1_hat = vae_model.decoder(h1)

    x0_hat = x0_hat.detach().cpu().numpy().reshape(num_timesteps, num_neurons)
    x1_hat = x1_hat.detach().cpu().numpy().reshape(num_timesteps, num_neurons)

    _plot_vae_samples(x0_in, x0_hat, x1_in, x1_hat, save_dir, epoch)
