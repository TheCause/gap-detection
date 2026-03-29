"""Deterministic Autoencoder for gap detection experiment.

Same architecture as the VAE but without stochastic layer and KL divergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder (identical conv layers to VAE)
        self.enc_conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)   # 28->14
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 14->7
        self.enc_fc = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)  # single deterministic bottleneck

        # Decoder (identical to VAE)
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, 64 * 7 * 7)
        self.dec_conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.view(h.size(0), -1)
        h = F.relu(self.enc_fc(h))
        return self.fc_z(h)

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        h = h.view(h.size(0), 64, 7, 7)
        h = F.relu(self.dec_conv1(h))
        return torch.sigmoid(self.dec_conv2(h))

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def ae_loss(recon_x, x):
    """BCE reconstruction loss only (no KL)."""
    return F.binary_cross_entropy(recon_x, x, reduction='sum')
