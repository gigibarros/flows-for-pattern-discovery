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

def plot_losses(train_losses, valid_losses, save_dir):
    """Plot training and validation losses."""
    plt.plot(train_losses, label="Training loss")
    plt.plot(valid_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "losses.png")
    plt.close()

def sample_p_xy(tarflow_model, sample_dir, config, epoch=None):
    samples_per_class = 4

    device = config["device"]["type"]

    input_dim = config["vae"]["input_dim"]
    token_size = config["tarflow_model"]["token_size"]

    zs = torch.randn(samples_per_class, input_dim, token_size)
    zs = zs.to(device)  

    for y in [0, 1]:
        with torch.no_grad():
            x_samples = tarflow_model.reverse(zs, y)   # xc_samples shape : (samples_per_class, input_dims["xc"], token_size)
    
        x_samples = x_samples.squeeze(-1)
        x_samples = x_samples.view(x_samples.size(0), 8, 8)
        x_samples = x_samples.detach().cpu()
        
        for j, sample in enumerate(x_samples):
            plt.figure()
            for series in sample:
                plt.plot(series.numpy())

            plt.title(f"hidden_var={y}, sample #{j}")

            if epoch is not None:
                fname = sample_dir / f"epoch_{epoch}_sample_{j}_hidden_{y}.png"
            else:
                fname = sample_dir / f"sample_{j}_hidden_{y}.png"

            plt.savefig(fname)
            plt.close()    
            
        del x_samples

    torch.cuda.empty_cache()