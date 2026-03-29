"""Training loop and latent extraction."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig
from vae import VAE, vae_loss
from utils import select_device, load_dataset, extract_latents as _extract


def get_filtered_dataset(excluded_classes: list, train: bool = True, data_root: str = "./data", dataset_name: str = "mnist"):
    """Load MNIST or FashionMNIST filtered by excluding specified classes."""
    return load_dataset(dataset_name=dataset_name, train=train,
                        excluded_classes=excluded_classes, data_root=data_root)


def train_vae(config: ExperimentConfig, condition: str, seed: int, device: str = None):
    """Train a VAE for one condition/seed."""
    if device is None:
        device = select_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    excluded = config.conditions[condition]
    dataset = load_dataset(config.dataset_name, train=True, excluded_classes=excluded)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    model = VAE(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss, _, _ = vae_loss(recon, batch_x, mu, logvar, beta=config.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(dataset)
            print(f"  [{condition}/seed{seed}] Epoch {epoch+1}/{config.epochs} - Loss: {avg:.2f}")

    # Save model
    run = config.run_name(condition, seed)
    torch.save(model.state_dict(), config.models_dir / f"{run}.pt")

    # Extract latents from test set
    latents, labels = extract_latents(model, config, excluded, device)
    np.save(config.latents_dir / f"{run}_latents.npy", latents)
    np.save(config.latents_dir / f"{run}_labels.npy", labels)

    print(f"  [{condition}/seed{seed}] Done. Latents: {latents.shape}")
    return model, latents, labels


def extract_latents(model, config: ExperimentConfig, excluded_classes: list, device: str = None):
    """Extract latent mu from test set."""
    if device is None:
        device = select_device()
    test_dataset = load_dataset(config.dataset_name, train=False, excluded_classes=excluded_classes)
    loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return _extract(model, loader, device)
