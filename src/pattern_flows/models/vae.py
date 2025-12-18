import torch
from torch import nn

class ResidualMLPBlock(nn.Module):
    """
    x -> x + F(x), where F is a small MLP.
    """
    def __init__(self, dim, inner_dim=None, dropout=0.0):
        super().__init__()
        inner_dim = inner_dim or dim
        self.act = nn.LeakyReLU(0.2)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            self.act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, width, out_dim, n_blocks=3, block_inner_dim=None, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[
            ResidualMLPBlock(width, inner_dim=block_inner_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.act = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(width, out_dim)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.act(x)
        x = self.blocks(x)
        x = self.act(x)
        x = self.out_proj(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_blocks=3, block_inner_dim=None, dropout=0.0):
        super().__init__()
        self.backbone = ResidualMLP(
            in_dim=input_dim,
            width=hidden_dim,
            out_dim=hidden_dim,
            n_blocks=n_blocks,
            block_inner_dim=block_inner_dim,
            dropout=dropout,
        )
        self.fc_mean   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        return self.fc_mean(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_blocks=3, block_inner_dim=None, dropout=0.0):
        super().__init__()
        self.net = ResidualMLP(
            in_dim=latent_dim,
            width=hidden_dim,
            out_dim=output_dim,
            n_blocks=n_blocks,
            block_inner_dim=block_inner_dim,
            dropout=dropout,
        )

    def forward(self, z):
        return self.net(z)

class VAE(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, device:torch.device, seed=None):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
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
    
class JointVAE(nn.Module):
    def __init__(self, encoders:dict[str, nn.Module], decoders:dict[str, nn.Module], device:torch.device, seed=None):
        super(JointVAE, self).__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.device = device
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(std.shape, generator=self.generator, device=self.device)
        z = mean + std * eps

        return z
    
    def forward(self, x, input_mod, target_mod):
        mean, log_var = self.encoders[input_mod](x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoders[target_mod](z)

        return x_hat, mean, log_var

def get_vae(config, multimodal=False, ckpt_file=None):
    """Initialize or load existing VAE model."""
    device     = config["device"]["type"]
    input_dim  = config["vae"]["input_dim"]
    hidden_dim = config["vae"]["hidden_dim"]
    latent_dim = config["vae"]["latent_dim"]

    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)

    if multimodal:
        # TO-DO : Add support for multimodal dataset
        vae = JointVAE(encoders=encoder, decoders=decoder, device=device)
    else:
        vae = VAE(encoder=encoder, decoder=decoder, device=device)

    if ckpt_file:
        vae.load_state_dict(torch.load(ckpt_file))
    
    return vae