import yaml
from pathlib import Path
import torch
import matplotlib.pyplot as plt


def load_config(path: str) -> dict:
    """Load config file as a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_ckpt_file(config, model_name, epoch=None):
    """Load checkpoint file."""
    dir = Path(config["paths"]["output_dir"])
    dir.mkdir(exist_ok=True, parents=True)

    if epoch:
        return dir / f"{model_name}_epoch_{epoch}.pth"
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
    plt.set_ylim(0, min(train_losses[0], 10))
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / f"losses_{save_tag}.png")
    plt.close()

def _plot_samples(xs, y, save_dir, save_tag=""):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    vmin = min(x.min() for x in xs)
    vmax = max(x.max() for x in xs)

    for idx, ax in enumerate(axs.flat):
        im = ax.imshow(xs[idx].T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Activity")
        ax.set_xlabel("Time")

    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.suptitle(f"Generative samples conditioned on y = {y}")

    fig.savefig(save_dir / f"samples_{save_tag}_y={y}.png")
    plt.close(fig)

def sample(vae, tarflow, config, save_dir, save_tag=""):
    num_timesteps     = config["data"]["num_timesteps"]
    num_neurons       = config["data"]["num_neurons"]
    token_size        = config["tarflow"]["token_size"]
    z_dim             = config["tarflow"]["z_dim"]
    device            = config["device"]["type"]
    samples_per_y     = 4

    samples = torch.randn(samples_per_y, z_dim, token_size).to(device)

    for y in [0, 1]:
        xs = []

        with torch.no_grad():
            zs = tarflow.reverse(samples, y)  # shape : (samples_per_class, z_dim, token_size)
            
        for z in zs:
            z = z.squeeze(-1)  # shape : (z_dim,)

            with torch.no_grad():
                x = vae.decoder(z)  # shape : (input_dim,)

            x = x.view(num_timesteps, num_neurons)  # shape : (num_timesteps, num_neurons)
            xs.append(x.detach().cpu().numpy())

        _plot_samples(xs, y, save_dir, save_tag)