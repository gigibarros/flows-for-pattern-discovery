import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.fc_input1 = nn.Linear(input_dim, hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h = self.LeakyReLU(self.fc_input1(x))
        h = self.LeakyReLU(self.fc_input2(h))
        mean = self.fc_mean(h)
        log_var = self.fc_logvar(h)

        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc_hidden1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.fc_hidden1(x))
        h = self.LeakyReLU(self.fc_hidden2(h))

        x_hat = self.fc_output(h)

        return x_hat

class MLPVAE(nn.Module):
    def __init__(self, cfg, seed=None):
        super(MLPVAE, self).__init__()

        device     = cfg["device"]["type"]
        input_dim  = cfg["vae"]["input_dim"]
        hidden_dim = cfg["vae"]["hidden_dim"]
        latent_dim = cfg["vae"]["latent_dim"]

        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
        self.device = device
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(std.shape, generator=self.generator, device=self.device)
        z = mean + std * eps

        return z
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var

class Conv2DVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_neurons   = cfg["data"]["num_neurons"]
        self.num_timesteps = cfg["data"]["num_timesteps"]
        self.latent_dim    = cfg["vae"]["latent_dim"]

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, T/2, N/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, T/4, N/4)
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.num_timesteps, self.num_neurons)
            h = self.encoder_conv(dummy)
            self._enc_out_shape = h.shape  # (1, C, H, W)
            enc_flat_dim = h.numel()       # 1 * C * H * W

        self.fc_mu    = nn.Linear(enc_flat_dim, self.latent_dim)
        self.fc_logvar= nn.Linear(enc_flat_dim, self.latent_dim)

        self.fc_decode = nn.Linear(self.latent_dim, enc_flat_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )

    def encode(self, x):
        assert x.dim() == 3, f"Expected (B, T, N), got {x.shape}"
        B = x.shape[0]

        # Add channel dimension: (B,1,T,N)
        x = x.view(B, 1, self.num_timesteps, self.num_neurons)

        h = self.encoder_conv(x)  # (B, C, H, W)
        h_flat = h.view(B, -1)

        mu = self.fc_mu(h_flat)
        log_var = self.fc_logvar(h_flat)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode(self, z):
        B = z.size(0)
        h_flat = self.fc_decode(z)

        _, C, H, W = self._enc_out_shape
        h = h_flat.view(B, C, H, W)     # (B, C, H, W)

        x_hat = self.decoder_conv(h)    # (B, 1, T, N)
        x_hat = x_hat.view(B, self.num_timesteps, self.num_neurons)
        return x_hat

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var
    
def get_vae(cfg, ckpt_file=None):
    """Initialize or load existing VAE model."""

    arch = cfg["vae"].get("arch", "mlp")

    if arch == "mlp":
        vae = MLPVAE(cfg)
    elif arch == "conv2d":
        vae = Conv2DVAE(cfg)
    else:
        raise ValueError(f"Unknown VAE arch: {arch}")

    if ckpt_file:
        vae.load_state_dict(torch.load(ckpt_file))
    
    return vae