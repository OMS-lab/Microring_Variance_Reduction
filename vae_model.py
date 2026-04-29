import torch
import torch.nn as nn

class VAE(nn.Module):
    """Variational Autoencoder: compress masks to `latent_dim` and reconstruct. 4 conv down -> (mu, logvar) -> 4 deconv up. """
    def __init__(self, latent_dim, image_size=128):
        super().__init__()
        self.image_size = int(image_size)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1,   32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32,  64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128,256, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )

        hw = self.image_size // 16
        flatten_size = 256 * hw * hw
        self.fc_mu     = nn.Linear(flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(flatten_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,  1,   kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        hw = self.image_size // 16
        h = h.view(-1, 256, hw, hw)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar
