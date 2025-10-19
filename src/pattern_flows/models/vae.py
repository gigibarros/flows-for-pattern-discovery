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
    input_dim = config.input_dim
    hidden_dim = config.hidden_dim
    latent_dim = config.latent_dim

    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, input_dim=input_dim)

    if multimodal:
        vae = JointVAE(encoders=encoder, decoders=decoder, device=config.device).to(config.device)
    else:
        vae = VAE(encoder=encoder, decoder=decoder, device=config.device).to(config.device)

    if ckpt_file:
        vae.load_state_dict(torch.load(ckpt_file))
    
    return vae