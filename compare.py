"""Statistical comparison between conditions using bootstrap."""

import numpy as np
from tda import compute_persistence, betti_numbers, persistence_features, compute_wasserstein


def bootstrap_compare(latents_a: np.ndarray, latents_b: np.ndarray,
                      n_bootstrap: int = 100, n_samples: int = 2000,
                      max_dim: int = 1) -> dict:
    """Bootstrap comparison of TDA features between two latent spaces.

    Returns statistics with confidence intervals.
    """
    wasserstein_h0 = []
    wasserstein_h1 = []
    betti_diff_h0 = []
    betti_diff_h1 = []
    life_diff_h1 = []

    for i in range(n_bootstrap):
        seed = 1000 + i
        res_a = compute_persistence(latents_a, max_dim=max_dim, n_samples=n_samples, seed=seed)
        res_b = compute_persistence(latents_b, max_dim=max_dim, n_samples=n_samples, seed=seed)

        # Wasserstein distances
        wd = compute_wasserstein(res_a["diagrams"], res_b["diagrams"])
        wasserstein_h0.append(wd["H0"])
        wasserstein_h1.append(wd["H1"])

        # Betti number differences
        betti_a = betti_numbers(res_a["diagrams"])
        betti_b = betti_numbers(res_b["diagrams"])
        betti_diff_h0.append(betti_b["H0"] - betti_a["H0"])
        betti_diff_h1.append(betti_b["H1"] - betti_a["H1"])

        # Persistence feature differences
        feat_a = persistence_features(res_a["diagrams"])
        feat_b = persistence_features(res_b["diagrams"])
        life_diff_h1.append(feat_b.get("H1_total_persistence", 0) - feat_a.get("H1_total_persistence", 0))

        if (i + 1) % 25 == 0:
            print(f"    Bootstrap {i+1}/{n_bootstrap}")

    def ci95(arr):
        arr = np.array(arr)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "p_gt_zero": float(np.mean(arr > 0)),
        }

    return {
        "wasserstein_H0": ci95(wasserstein_h0),
        "wasserstein_H1": ci95(wasserstein_h1),
        "betti_diff_H0": ci95(betti_diff_h0),
        "betti_diff_H1": ci95(betti_diff_h1),
        "persistence_diff_H1": ci95(life_diff_h1),
        "n_bootstrap": n_bootstrap,
    }


def null_distribution(latents: np.ndarray, n_bootstrap: int = 100,
                      n_samples: int = 2000, max_dim: int = 1) -> dict:
    """Null distribution: bootstrap within same latent space (split in half).

    This gives us the expected variation when there is NO gap.
    """
    wasserstein_h0 = []
    wasserstein_h1 = []

    n = len(latents)
    for i in range(n_bootstrap):
        rng = np.random.RandomState(2000 + i)
        perm = rng.permutation(n)
        half = n // 2
        split_a, split_b = latents[perm[:half]], latents[perm[half:2*half]]

        res_a = compute_persistence(split_a, max_dim=max_dim, n_samples=min(n_samples, half), seed=2000+i)
        res_b = compute_persistence(split_b, max_dim=max_dim, n_samples=min(n_samples, half), seed=3000+i)

        wd = compute_wasserstein(res_a["diagrams"], res_b["diagrams"])
        wasserstein_h0.append(wd["H0"])
        wasserstein_h1.append(wd["H1"])

        if (i + 1) % 25 == 0:
            print(f"    Null bootstrap {i+1}/{n_bootstrap}")

    def ci95(arr):
        arr = np.array(arr)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_high_95": float(np.percentile(arr, 95)),
            "ci_high_99": float(np.percentile(arr, 99)),
        }

    return {
        "null_wasserstein_H0": ci95(wasserstein_h0),
        "null_wasserstein_H1": ci95(wasserstein_h1),
        "n_bootstrap": n_bootstrap,
    }
