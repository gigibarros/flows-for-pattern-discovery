import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import umap.umap_ as umap
import numpy as np
import seaborn as sns

def load_yaml(path: str) -> dict:
    """Load a YAML file as a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Recursively merge dict2 into dict1."""
    result = dict1.copy()
    for k, v in dict2.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def load_config(base_path: str, model_path: str) -> dict:
    """Load and merge base and model-specific config."""
    base = load_yaml(base_path)
    model = load_yaml(model_path)
    return deep_merge(base, model)

def get_ckpt_file(config, model_name, epoch=None):
    """Load checkpoint file."""
    dir = Path(config.output_dir) / f"{model_name}"
    dir.mkdir(exist_ok=True, parents=True)

    if epoch:
        return dir / f"{model_name}_epoch_{epoch}.pth"
    else:
        return dir / f"{model_name}.pth"

def get_save_dir(output_path, model_name):
    """Load save directory."""
    dir = Path(output_path) / f"{model_name}"
    dir.mkdir(exist_ok=True, parents=True)

    return dir

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

def vae_umap(vae_model, valid_loader, save_dir=None):
    """Plot UMAP of VAE latent space."""
    zs = []
    labels = []

    vae_model.eval()

    with torch.no_grad():
        for batch in valid_loader:
            x = batch[0]
            y = batch[1]

            x = torch.flatten(x, start_dim=1, end_dim=-1).to(vae_model.device)

            _, mean, _ = vae_model(x)

            zs.append(mean.cpu().numpy())
            labels.append(y.cpu().numpy())

    zs = np.concatenate(zs, axis=0)
    labels = np.concatenate(labels, axis=0)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(zs)

    plt.figure(figsize=(8, 6))

    colors = sns.color_palette("Set1", n_colors=len(np.unique(labels)))

    for i, label_val in enumerate(np.unique(labels)):
        idx = labels == label_val
        plt.scatter(
            embedding[idx, 0], embedding[idx, 1],
            c=[colors[i]], label=f"Class {label_val}", s=12, alpha=0.8
        )    
        
    plt.title("UMAP projection of VAE latent space")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / "vae_umap.png")
        plt.close()
    else:
        plt.show()