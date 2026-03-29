"""Shared utilities for gap detection experiments."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def select_device():
    """Auto-detect best device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dataset(dataset_name="mnist", train=True, excluded_classes=None, data_root="./data"):
    """Load MNIST or FashionMNIST, optionally filtering out classes."""
    ds_cls = datasets.FashionMNIST if dataset_name == "fashion_mnist" else datasets.MNIST
    ds = ds_cls(root=data_root, train=train, download=True, transform=transforms.ToTensor())
    if not excluded_classes:
        return ds
    targets = ds.targets if isinstance(ds.targets, torch.Tensor) else torch.tensor(ds.targets)
    mask = torch.tensor([int(t) not in excluded_classes for t in targets])
    return Subset(ds, mask.nonzero(as_tuple=True)[0].tolist())


def load_dataset_flat(dataset_name="mnist", train=False, excluded_classes=None, data_root="./data"):
    """Load dataset as flat numpy vectors (784-dim, for PCA)."""
    ds_cls = datasets.FashionMNIST if dataset_name == "fashion_mnist" else datasets.MNIST
    ds = ds_cls(root=data_root, train=train, download=True, transform=transforms.ToTensor())
    X = ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    targets = ds.targets if isinstance(ds.targets, torch.Tensor) else torch.tensor(ds.targets)
    y = targets.numpy()
    if excluded_classes:
        mask = np.array([int(t) not in excluded_classes for t in y])
        X, y = X[mask], y[mask]
    return X, y


def train_model(model, dataset, config, device, label="", beta=None):
    """Generic training loop for VAE or AE.

    Detects model type via hasattr(model, 'reparameterize').
    """
    from vae import vae_loss
    from ae import ae_loss as ae_loss_fn

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    is_vae = hasattr(model, 'reparameterize')
    b = beta if beta is not None else config.beta

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            if is_vae:
                recon, mu, logvar = model(batch_x)
                loss, _, _ = vae_loss(recon, batch_x, mu, logvar, beta=b)
            else:
                recon, z = model(batch_x)
                loss = ae_loss_fn(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(dataset)
            print(f"  [{label}] Epoch {epoch+1}/{config.epochs} - Loss: {avg:.2f}")
    return model


def extract_latents(model, dataloader, device):
    """Extract latent vectors from a dataloader. Handles VAE (mu) and AE (z)."""
    model.eval()
    all_z, all_labels = [], []
    is_vae = hasattr(model, 'reparameterize')
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            if is_vae:
                z, _ = model.encode(batch_x)
            else:
                z = model.encode(batch_x)
            all_z.append(z.cpu().numpy())
            all_labels.append(batch_y.numpy())
    return np.concatenate(all_z), np.concatenate(all_labels)


def compute_centroids(Z, labels, classes=None):
    """Per-class centroids from latent vectors and labels."""
    if classes is None:
        classes = sorted(set(int(x) for x in labels))
    return {int(c): Z[labels == c].mean(axis=0) for c in classes if (labels == c).sum() > 0}


def procrustes_align(X_ref, X_tgt):
    """Procrustes: align X_tgt to X_ref (rotation + scaling + translation).

    Returns: (X_aligned, R, s, t) such that X_aligned = s * X_tgt @ R + t.
    """
    mu_r = X_ref.mean(axis=0)
    mu_t = X_tgt.mean(axis=0)
    Xr = X_ref - mu_r
    Xt = X_tgt - mu_t
    U, S, Vt = np.linalg.svd(Xr.T @ Xt)
    d = np.linalg.det(U @ Vt)
    D = np.eye(len(S))
    D[-1, -1] = np.sign(d)
    R = Vt.T @ D @ U.T
    Xt_rot = Xt @ R
    s = np.trace(Xr.T @ Xt_rot) / np.trace(Xt_rot.T @ Xt_rot)
    t = mu_r - s * (mu_t @ R)
    X_aligned = s * X_tgt @ R + t
    return X_aligned, R, s, t


def normalize_latents(Z, n_sample=1000, seed=42):
    """Normalize latents: center + median pairwise L2 scaling."""
    from scipy.spatial.distance import pdist
    Z_c = Z - Z.mean(axis=0)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Z_c), min(n_sample, len(Z_c)), replace=False)
    s = np.median(pdist(Z_c[idx]))
    return Z_c / s, s
