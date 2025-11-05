import torch
from torch.optim import Adam
import torch.nn.functional as F

import pattern_flows.utils as u
from pattern_flows.data.toy_data import get_train_loader, get_valid_loader
from pattern_flows.models.vae import get_vae

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
import seaborn as sns

def compute_loss(x):
    x_hat, mean, log_var = vae_model(x)

    reconst_loss = F.mse_loss(x_hat, x, reduction='mean')
    d_kl = -0.5 * torch.sum( 1 + log_var - mean.pow(2) - log_var.exp() )    

    return reconst_loss, d_kl

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

if __name__ == "__main__":
    cfg = u.load_config("configs/base.yaml")
    
    # Load parameters
    # TO-DO : Create helper function for loading parameters
    device        = cfg["device"]["type"]
    batch_size    = cfg["training"]["batch_size"]
    output_path   = cfg["paths"]["output_path"]

    num_samples   = cfg["data"]["num_samples"]
    num_neurons   = cfg["data"]["num_neurons"]
    num_words     = cfg["data"]["num_words"]
    num_timesteps = cfg["data"]["num_timesteps"]

    model_name    = cfg["vae_model"]["name"]
    input_dim     = cfg["vae_model"]["input_dim"]
    hidden_dim    = cfg["vae_model"]["hidden_dim"]
    latent_dim    = cfg["vae_model"]["latent_dim"]
    num_epochs    = cfg["vae_model"]["num_epochs"]
    beta_loss     = cfg["vae_model"]["beta_loss"]
    lr            = cfg["vae_model"]["lr"]

    train_loader = get_train_loader(cfg)
    valid_loader = get_valid_loader(cfg)

    vae_model = get_vae(cfg, device, multimodal=False)

    optimizer = Adam(vae_model.parameters(), lr=lr)

    save_dir = u.get_save_dir(__file__, output_path, model_name)
    save_dir.mkdir(exist_ok=True, parents=True)

    train_losses = []
    valid_losses = []

    self_modal_losses = []
    cross_modal_losses = []

    reconst_losses = []
    kl_losses = []

    print("Start training VAE...")

    for epoch in tqdm(range(num_epochs)):
        total_train_loss = 0
        total_valid_loss = 0

        total_reconst_loss = 0 
        total_kl_loss = 0

        vae_model.train()

        for batch_idx, batch in enumerate(train_loader):
            x = batch[0]
            x = torch.flatten(x, start_dim=1, end_dim=-1).to(device)

            optimizer.zero_grad()

            # Compute losses
            reconst_loss, kl_loss = compute_loss(x)
            loss = reconst_loss + beta_loss * kl_loss

            total_reconst_loss += reconst_loss.item()
            total_kl_loss += kl_loss.item()

            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        reconst_losses.append( total_reconst_loss / ((batch_idx + 1) * batch_size) )
        kl_losses.append( total_kl_loss / ((batch_idx + 1) * batch_size) )  

        train_losses.append( total_train_loss / ((batch_idx + 1) * batch_size) )

        vae_model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                x = batch[0]
                x = torch.flatten(x, start_dim=1, end_dim=-1).to(device)
                
                optimizer.zero_grad()

                # Compute losses
                reconst_loss, kl_loss = compute_loss(x)

                loss = reconst_loss + beta_loss * kl_loss

                total_valid_loss += loss.item()  

            valid_losses.append( total_valid_loss / ((batch_idx + 1) * batch_size) )

        tqdm.write(f"\tepoch {epoch + 1} complete")
        tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
        tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")
        
    print("training complete")
    torch.save(vae_model.state_dict(), save_dir / f"{model_name}.pth")

    u.plot_losses(train_losses, valid_losses, save_dir)
    vae_umap(vae_model, valid_loader, save_dir)