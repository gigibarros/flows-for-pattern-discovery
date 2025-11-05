import pattern_flows.utils as u
from pattern_flows.data.toy_data import get_toy_dataset, get_train_loader, get_valid_loader
from pattern_flows.models.vae import get_vae
from pattern_flows.external.ml_tarflow.tarflow.model import get_tarflow_model

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

if __name__ == "__main__":
    cfg = u.load_config("src/pattern_flows/configs/base.yaml")
    
    # Load parameters
    # TO-DO : Create helper function for loading parameters
    device        = cfg["device"]["type"]
    batch_size    = cfg["training"]["batch_size"]
    output_path   = cfg["paths"]["output_path"]
    
    vae_model_name = cfg["vae"]["name"]
    input_dim      = cfg["vae"]["input_dim"]
    mid_dim        = cfg["vae"]["hidden_dim"]
    h_dim          = cfg["vae"]["latent_dim"]

    tarflow_model_name = cfg["tarflow"]["name"]
    token_size         = cfg["tarflow"]["token_size"]
    num_classes        = cfg["tarflow"]["num_classes"]
    num_epochs         = cfg["tarflow"]["num_epochs"]
    lr                 = cfg["tarflow"]["lr"]
    weight_decay       = cfg["tarflow"]["weight_decay"]
    noise_std          = cfg["tarflow"]["noise_std"]
    sample_freq        = cfg["tarflow"]["sample_freq"]

    vae_model = get_vae(cfg, device, ckpt_file=u.get_ckpt_file(cfg, vae_model_name))
    
    # tarflow_model = get_tarflow_model(cfg, input_dim, num_classes=num_classes)
    try:
        tarflow_model = get_tarflow_model(cfg, input_dim, num_classes=num_classes)
    except Exception as e:
        import traceback
        print("Failed to create tarflow model:")
        traceback.print_exc()
        raise


    valid_dataset = get_toy_dataset(cfg)

    train_loader = get_train_loader(cfg)
    valid_loader = get_valid_loader(cfg)

    print("data loaders created")

    save_dir = u.get_save_dir(__file__, output_path, tarflow_model_name)
    sample_dir = save_dir / "_samples"
    ckpt_dir = save_dir / "checkpoints"

    save_dir.mkdir(exist_ok=True, parents=True)
    sample_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    print("directories created")

    optimizer = torch.optim.AdamW(tarflow_model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=weight_decay)
    lr_schedule = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(num_epochs)):
        total_train_loss = 0
        total_valid_loss = 0

        tarflow_model.train()

        print("start training tarflow...")

        for batch_idx, batch in enumerate(train_loader):
                x = batch[0]                # shape : (batch_size, input_dim)
                y = batch[1]                # shape : (batch_size)

                x = x.view(x.size(0), input_dim, token_size).to(device)   # shape : (batch_size, num_tokens, token_size)
                y = y.to(device)

                eps = noise_std * torch.randn_like(x)
                x = x + eps
                x = x.to(device)

                optimizer.zero_grad()

                z, outputs, logdets = tarflow_model(x, y)
                loss = tarflow_model.get_loss(z, logdets)

                total_train_loss += loss.item()

                loss.backward()
                optimizer.step()
                lr_schedule.step()

        train_losses.append( total_train_loss / ((batch_idx + 1) * batch_size) )

        tarflow_model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                    x = batch[0]                # shape : (batch_size, input_dims)
                    y = batch[1]                # shape : (batch_size)

                    x = x.view(x.size(0), input_dim, token_size)   # shape : (batch_size, num_tokens, token_size)
                    y = y.to(device)

                    eps = noise_std * torch.randn_like(x).to(device)
                    x = x + eps

                    z, outputs, logdets = tarflow_model(x, y)
                    loss = tarflow_model.get_loss(z, logdets)

                    total_valid_loss += loss.item()

        valid_losses.append( total_valid_loss / ((batch_idx + 1) * batch_size) )

        tqdm.write(f"\tepoch {epoch} complete")
        tqdm.write(f"\taverage training loss: {train_losses[epoch]}")
        tqdm.write(f"\taverage validation loss: {valid_losses[epoch]}")

        if (epoch + 1) % sample_freq == 0 or epoch == 0 or epoch == 99:
            u.sample_p_xy(tarflow_model, sample_dir, epoch=epoch)
            torch.save(tarflow_model.state_dict(), ckpt_dir / f"{tarflow_model_name}_epoch_{epoch}.pth")
            tqdm.write(f"saved and sampled at epoch {epoch + 1}")

    print("training complete")
    u.plot_losses(train_losses, valid_losses, save_dir)