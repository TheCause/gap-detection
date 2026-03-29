"""Persistent homology computations using ripser."""

import numpy as np
from ripser import ripser
from persim import wasserstein as wasserstein_distance


def compute_persistence(latents: np.ndarray, max_dim: int = 1, n_samples: int = 2000,
                        seed: int = 42) -> dict:
    """Compute persistent homology on subsampled latent space.

    Returns dict with 'diagrams' (list of arrays per dim) and 'metadata'.
    """
    rng = np.random.RandomState(seed)
    if len(latents) > n_samples:
        idx = rng.choice(len(latents), n_samples, replace=False)
        points = latents[idx]
    else:
        points = latents

    result = ripser(points, maxdim=max_dim)
    return {
        "diagrams": result["dgms"],
        "n_points": len(points),
    }


def betti_numbers(diagrams: list, threshold: float = None) -> dict:
    """Count Betti numbers from persistence diagrams.

    If threshold is None, count features that persist (death > birth).
    If threshold given, count features alive at that scale.
    """
    betti = {}
    for dim, dgm in enumerate(diagrams):
        if threshold is None:
            # Count all finite features
            finite = dgm[dgm[:, 1] < np.inf]
            betti[f"H{dim}"] = len(finite)
        else:
            alive = np.sum((dgm[:, 0] <= threshold) & (dgm[:, 1] > threshold))
            betti[f"H{dim}"] = int(alive)
    return betti


def persistence_features(diagrams: list) -> dict:
    """Extract statistical features from persistence diagrams."""
    features = {}
    for dim, dgm in enumerate(diagrams):
        finite = dgm[dgm[:, 1] < np.inf]
        if len(finite) == 0:
            features[f"H{dim}_count"] = 0
            features[f"H{dim}_mean_life"] = 0.0
            features[f"H{dim}_max_life"] = 0.0
            features[f"H{dim}_total_persistence"] = 0.0
            continue

        lifetimes = finite[:, 1] - finite[:, 0]
        features[f"H{dim}_count"] = len(finite)
        features[f"H{dim}_mean_life"] = float(np.mean(lifetimes))
        features[f"H{dim}_max_life"] = float(np.max(lifetimes))
        features[f"H{dim}_total_persistence"] = float(np.sum(lifetimes))
        features[f"H{dim}_std_life"] = float(np.std(lifetimes))

        # Top-k longest features
        top_k = min(5, len(lifetimes))
        top_lifetimes = np.sort(lifetimes)[-top_k:][::-1]
        features[f"H{dim}_top{top_k}_lifetimes"] = top_lifetimes.tolist()

    return features


def compute_wasserstein(diag1: list, diag2: list) -> dict:
    """Wasserstein distance between two persistence diagrams, per dimension."""
    distances = {}
    for dim in range(min(len(diag1), len(diag2))):
        d = wasserstein_distance(diag1[dim], diag2[dim])
        distances[f"H{dim}"] = float(d)
    return distances
