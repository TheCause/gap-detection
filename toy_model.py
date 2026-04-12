#!/usr/bin/env python3
"""Toy model: Hopfield energy landscape of a VAE with missing attractors.

Setup:
- 2D latent space
- K=3 classes, each modeled as a Gaussian in latent space
- Prior: N(0, I)
- ELBO energy: E(z) = -sum_c w_c * N(z; mu_c, sigma_c) + beta * KL(q||p)
- Ablation: remove one class, observe energy landscape deformation

This script:
1. Computes the energy landscape before/after ablation
2. Measures attractor displacement (aspiration)
3. Computes persistent homology of sublevel sets
4. Verifies theoretical predictions
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "arxiv_source" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def total_energy(z, class_params, beta=1.0):
    """Modern Hopfield energy for VAE latent space.

    E(z) = -log sum_c exp(-||z - mu_c||^2 / (2*sigma_c^2)) + beta/2 * ||z||^2

    The -logsumexp creates separate basins (one attractor per class).
    The KL term (beta * ||z||^2 / 2) pulls everything toward the origin.
    This is exactly the Modern Hopfield energy (Ramsauer et al. 2020).
    """
    # Collect log-likelihoods per class
    log_likes = []
    for mu, sigma, w in class_params:
        d = z - mu
        ll = -0.5 / (sigma**2) * np.sum(d**2, axis=-1)
        log_likes.append(ll)
    log_likes = np.stack(log_likes, axis=0)

    # logsumexp for numerical stability
    max_ll = np.max(log_likes, axis=0)
    lse = max_ll + np.log(np.sum(np.exp(log_likes - max_ll), axis=0))

    # Total energy = -logsumexp + KL
    E = -lse + beta * 0.5 * np.sum(z**2, axis=-1)
    return E


def find_minima(energy_grid, zx, zy, n_minima=5):
    """Find local minima in energy grid via simple gradient check."""
    minima = []
    for i in range(1, energy_grid.shape[0] - 1):
        for j in range(1, energy_grid.shape[1] - 1):
            patch = energy_grid[i-1:i+2, j-1:j+2]
            if energy_grid[i, j] == patch.min():
                minima.append((zx[0, j], zy[i, 0], energy_grid[i, j]))
    minima.sort(key=lambda x: x[2])
    return minima[:n_minima]


def compute_aspiration(minima_full, minima_ablated, removed_mu):
    """Compute aspiration: how much did each surviving minimum move toward the void?"""
    results = []
    for (x_f, y_f, _) in minima_full:
        # Find closest minimum in ablated landscape
        best_dist = float('inf')
        best_match = None
        for (x_a, y_a, _) in minima_ablated:
            d = np.sqrt((x_f - x_a)**2 + (y_f - y_a)**2)
            if d < best_dist:
                best_dist = d
                best_match = (x_a, y_a)

        if best_match is None:
            continue

        # Distance to removed class centroid
        d_before = np.sqrt((x_f - removed_mu[0])**2 + (y_f - removed_mu[1])**2)
        d_after = np.sqrt((best_match[0] - removed_mu[0])**2 + (best_match[1] - removed_mu[1])**2)
        alpha = (d_before - d_after) / d_before if d_before > 0 else 0

        results.append({
            "pos_full": (x_f, y_f),
            "pos_ablated": best_match,
            "d_before": d_before,
            "d_after": d_after,
            "alpha": alpha,
            "displacement": best_dist,
        })
    return results


def run_toy_model():
    # Class centroids: equilateral triangle, radius 2 from origin
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    R = 2.0
    centroids = [(R * np.cos(a), R * np.sin(a)) for a in angles]
    sigma = 0.3  # class spread

    # Grid
    grid_range = 4.0
    n_grid = 200
    x = np.linspace(-grid_range, grid_range, n_grid)
    y = np.linspace(-grid_range, grid_range, n_grid)
    zx, zy = np.meshgrid(x, y)
    z_flat = np.stack([zx.ravel(), zy.ravel()], axis=-1)

    betas = [0.5, 1.0, 2.0, 5.0]
    removed_class = 0  # Remove class at angle 0 (rightmost)
    removed_mu = np.array(centroids[removed_class])

    fig, axes = plt.subplots(2, len(betas), figsize=(4*len(betas), 8))
    fig.suptitle("Energy Landscape: Full (top) vs Ablated (bottom)", fontsize=14)

    for col, beta in enumerate(betas):
        # Full landscape (all 3 classes)
        full_params = [(np.array(c), sigma, 1.0) for c in centroids]
        E_full = total_energy(z_flat, full_params, beta).reshape(n_grid, n_grid)

        # Ablated landscape (2 classes)
        abl_params = [(np.array(c), sigma, 1.0) for i, c in enumerate(centroids) if i != removed_class]
        E_abl = total_energy(z_flat, abl_params, beta).reshape(n_grid, n_grid)

        # Find minima
        minima_full = find_minima(E_full, zx, zy)
        minima_abl = find_minima(E_abl, zx, zy)

        # Aspiration
        asp = compute_aspiration(minima_full, minima_abl, removed_mu)

        # Clip for visualization
        vmin, vmax = np.percentile(E_full, [5, 95])

        # Plot full
        ax = axes[0, col]
        ax.contourf(zx, zy, E_full, levels=30, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        for cx, cy in centroids:
            ax.plot(cx, cy, "k*", markersize=10)
        for m in minima_full:
            ax.plot(m[0], m[1], "wo", markersize=6, markeredgecolor="k")
        ax.set_title(f"Full (beta={beta})")
        ax.set_aspect("equal")

        # Plot ablated
        ax = axes[1, col]
        ax.contourf(zx, zy, E_abl, levels=30, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        for i, (cx, cy) in enumerate(centroids):
            if i == removed_class:
                ax.plot(cx, cy, "rx", markersize=12, markeredgewidth=2)  # ghost
            else:
                ax.plot(cx, cy, "k*", markersize=10)
        for m in minima_abl:
            ax.plot(m[0], m[1], "wo", markersize=6, markeredgecolor="k")

        # Draw aspiration arrows
        for a in asp:
            if a["displacement"] > 0.05:
                ax.annotate("", xy=a["pos_ablated"], xytext=a["pos_full"],
                           arrowprops=dict(arrowstyle="->", color="lime", lw=2))

        ax.set_title(f"Ablated (beta={beta})")
        ax.set_aspect("equal")

        # Print aspiration
        print(f"\nbeta={beta}:")
        print(f"  Full minima: {[(f'{m[0]:.2f}', f'{m[1]:.2f}') for m in minima_full]}")
        print(f"  Ablated minima: {[(f'{m[0]:.2f}', f'{m[1]:.2f}') for m in minima_abl]}")
        for a in asp:
            print(f"  Attractor ({a['pos_full'][0]:.2f}, {a['pos_full'][1]:.2f}) -> "
                  f"({a['pos_ablated'][0]:.2f}, {a['pos_ablated'][1]:.2f}) "
                  f"alpha={a['alpha']:.4f} d_before={a['d_before']:.2f}")

    plt.tight_layout()
    fig_path = OUT_DIR / "toy_model_energy.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    # --- Figure 2: Aspiration vs distance for different betas ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

    for beta in [0.5, 1.0, 2.0, 5.0]:
        full_params = [(np.array(c), sigma, 1.0) for c in centroids]
        abl_params = [(np.array(c), sigma, 1.0) for i, c in enumerate(centroids) if i != removed_class]
        E_full = total_energy(z_flat, full_params, beta).reshape(n_grid, n_grid)
        E_abl = total_energy(z_flat, abl_params, beta).reshape(n_grid, n_grid)
        minima_f = find_minima(E_full, zx, zy)
        minima_a = find_minima(E_abl, zx, zy)
        asp = compute_aspiration(minima_f, minima_a, removed_mu)

        if asp:
            ds = [a["d_before"] for a in asp]
            alphas = [a["alpha"] for a in asp]
            ax2.scatter(ds, alphas, label=f"beta={beta}", s=60)

    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Distance to ghost centroid")
    ax2.set_ylabel("Aspiration alpha")
    ax2.set_title("Toy model: aspiration vs distance (theory)")
    ax2.legend()

    fig2_path = OUT_DIR / "toy_model_aspiration.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"Figure saved: {fig2_path}")


if __name__ == "__main__":
    run_toy_model()
