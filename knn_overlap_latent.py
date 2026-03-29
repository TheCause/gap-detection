"""
k-NN overlap on VAE latent vectors (dim=16).
Compares learned proximity structure vs pixel-space proximity.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def knn_indices_batched(X, k, batch_size=2000):
    """Compute k-NN indices using batched L2, pure numpy."""
    N = X.shape[0]
    X_sq = np.sum(X ** 2, axis=1)
    all_indices = np.empty((N, k), dtype=np.int64)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = X[start:end]
        batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)
        dists = batch_sq + X_sq[np.newaxis, :] - 2.0 * batch @ X.T

        for i in range(end - start):
            dists[i, start + i] = np.inf

        idx = np.argpartition(dists, k, axis=1)[:, :k]
        for i in range(end - start):
            sub_dists = dists[i, idx[i]]
            order = np.argsort(sub_dists)
            idx[i] = idx[i][order]

        all_indices[start:end] = idx

    return all_indices


def compute_overlap_matrix(X, y, k=50, n_classes=10):
    """10x10 overlap matrix from k-NN."""
    indices = knn_indices_batched(X, k)
    overlap = np.zeros((n_classes, n_classes))
    for c in range(n_classes):
        mask_c = y == c
        neighbor_labels = y[indices[mask_c]]
        for j in range(n_classes):
            overlap[c, j] = np.mean(neighbor_labels == j)
    return overlap


def run_latent_overlap():
    from config import ExperimentConfig
    cfg = ExperimentConfig()
    latents_dir = cfg.latents_dir
    output_dir = cfg.base_dir
    seeds = cfg.seeds
    k_values = [10, 30, 50, 100, 200]

    # Load and concatenate full-condition latents across seeds
    all_X, all_y = [], []
    for seed in seeds:
        X = np.load(latents_dir / f"full_seed{seed}_latents.npy")
        y = np.load(latents_dir / f"full_seed{seed}_labels.npy")
        all_X.append(X)
        all_y.append(y)
        print(f"  Seed {seed}: {X.shape}, labels {np.bincount(y.astype(int))}")

    # Use seed 42 as primary (avoid inflating N by concatenation)
    X_primary = all_X[0]
    y_primary = all_y[0].astype(int)
    print(f"\nPrimary latent space (seed 42): {X_primary.shape}")
    print(f"  Classes: {np.bincount(y_primary)}")

    # Also compute average across seeds for robustness
    results_per_seed = {}
    for si, seed in enumerate(seeds):
        X = all_X[si]
        y = all_y[si].astype(int)
        print(f"\n--- Seed {seed} ---")
        M = compute_overlap_matrix(X, y, k=50)
        results_per_seed[seed] = M

        sym_37 = (M[3, 7] + M[7, 3]) / 2
        sym_49 = (M[4, 9] + M[9, 4]) / 2
        sym_35 = (M[3, 5] + M[5, 3]) / 2
        print(f"  overlap(3,7) = {sym_37:.4f}")
        print(f"  overlap(4,9) = {sym_49:.4f}")
        print(f"  overlap(3,5) = {sym_35:.4f}")

    # Average matrix across seeds
    M_avg = np.mean([results_per_seed[s] for s in seeds], axis=0)
    print(f"\n=== AVERAGE ACROSS SEEDS (k=50) ===")

    # Rank all pairs
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            sym = (M_avg[i, j] + M_avg[j, i]) / 2
            pairs.append((i, j, sym))
    pairs.sort(key=lambda x: -x[2])

    print("\nTOP 15 MOST OVERLAPPING PAIRS (latent, k=50, avg 3 seeds):")
    for rank, (i, j, v) in enumerate(pairs[:15], 1):
        print(f"  {rank:>2}. ({i},{j}): {v:.4f}")

    # Find rank of (3,7)
    for rank, (i, j, v) in enumerate(pairs, 1):
        if (i, j) == (3, 7):
            print(f"\n>>> (3,7) rank: {rank}/45, overlap: {v:.4f}")

    # k sweep on primary seed
    print("\n=== K SWEEP (seed 42) ===")
    k_results = {}
    for k in k_values:
        M = compute_overlap_matrix(X_primary, y_primary, k=k)
        k_results[k] = M
        sym_37 = (M[3, 7] + M[7, 3]) / 2
        print(f"  k={k:>3d}: overlap(3,7) = {sym_37:.4f}")

    # Ranking stability
    print("\n=== RANKING STABILITY (top-5 per k, seed 42) ===")
    for k in k_values:
        M = k_results[k]
        p = []
        for i in range(10):
            for j in range(i + 1, 10):
                p.append((i, j, (M[i, j] + M[j, i]) / 2))
        p.sort(key=lambda x: -x[2])
        top5 = [(i, j) for i, j, _ in p[:5]]
        print(f"  k={k:>3d}: {top5}")

    # Compare pixel vs latent rankings
    print("\n=== PIXEL vs LATENT COMPARISON (k=50) ===")
    pixel_path = output_dir / "knn_overlap.json"
    if pixel_path.exists():
        with open(pixel_path) as f:
            pixel_data = json.load(f)
        M_pixel = np.array(pixel_data["results"]["50"]["matrix"])

        pixel_pairs = []
        for i in range(10):
            for j in range(i + 1, 10):
                pixel_pairs.append((i, j, (M_pixel[i, j] + M_pixel[j, i]) / 2))
        pixel_pairs.sort(key=lambda x: -x[2])
        pixel_rank = {(i, j): r for r, (i, j, _) in enumerate(pixel_pairs, 1)}

        latent_rank = {(i, j): r for r, (i, j, _) in enumerate(pairs, 1)}

        print(f"\n{'Pair':>8} | {'Pixel rank':>10} | {'Latent rank':>11} | {'Delta':>6} | {'Pixel ovlp':>10} | {'Latent ovlp':>11}")
        print("-" * 75)
        # Show biggest movers
        deltas = []
        for i in range(10):
            for j in range(i + 1, 10):
                pr = pixel_rank.get((i, j), 45)
                lr = latent_rank.get((i, j), 45)
                pv = (M_pixel[i, j] + M_pixel[j, i]) / 2
                lv = (M_avg[i, j] + M_avg[j, i]) / 2
                deltas.append((i, j, pr, lr, pr - lr, pv, lv))

        # Sort by biggest rank improvement (pixel_rank - latent_rank, positive = moved UP)
        deltas.sort(key=lambda x: -x[4])
        print("\nBIGGEST CLIMBERS (pixel → latent):")
        for i, j, pr, lr, d, pv, lv in deltas[:10]:
            arrow = "^" if d > 0 else "v" if d < 0 else "="
            print(f"  ({i},{j}): pixel #{pr:>2} → latent #{lr:>2}  ({arrow}{abs(d):>2})  "
                  f"pixel={pv:.4f} latent={lv:.4f}")

        print("\nBIGGEST FALLERS (pixel → latent):")
        deltas.sort(key=lambda x: x[4])
        for i, j, pr, lr, d, pv, lv in deltas[:10]:
            arrow = "^" if d > 0 else "v" if d < 0 else "="
            print(f"  ({i},{j}): pixel #{pr:>2} → latent #{lr:>2}  ({arrow}{abs(d):>2})  "
                  f"pixel={pv:.4f} latent={lv:.4f}")

    # Experiment conditions analysis
    print("\n=== EXPERIMENT CONDITIONS (latent, k=50, avg seeds) ===")
    conditions = {
        "minus_7": [7],
        "minus_3_7": [3, 7],
        "minus_cluster": [3, 5, 8],
    }
    for cond, excluded in conditions.items():
        remaining = [c for c in range(10) if c not in excluded]
        overlaps = []
        for ex in excluded:
            for rem in remaining:
                overlaps.append((M_avg[ex, rem] + M_avg[rem, ex]) / 2)
        internal = []
        for idx, ex1 in enumerate(excluded):
            for ex2 in excluded[idx+1:]:
                internal.append((M_avg[ex1, ex2] + M_avg[ex2, ex1]) / 2)
        int_ov = np.mean(internal) if internal else 0
        print(f"  {cond}: excl<->remain={np.mean(overlaps):.4f}, internal={int_ov:.4f}")

    # Save results
    save_data = {
        "space": "VAE_latent_dim16",
        "primary_k": 50,
        "seeds": seeds,
        "matrix_avg_k50": M_avg.tolist(),
        "top_pairs_k50": [(i, j, round(v, 6)) for i, j, v in pairs],
    }
    with open(output_dir / "knn_overlap_latent.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {output_dir / 'knn_overlap_latent.json'}")

    # Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, M, title in [
        (axes[0], M_pixel if pixel_path.exists() else M_avg, "Pixel space (k=50)"),
        (axes[1], M_avg, "VAE latent space (k=50, avg 3 seeds)"),
    ]:
        M_plot = M.copy()
        np.fill_diagonal(M_plot, np.nan)
        im = ax.imshow(M_plot, cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Neighbor class")
        ax.set_ylabel("Source class")
        ax.set_title(title)
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                val = M_plot[i, j]
                color = "white" if val > 0.04 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color=color)
        plt.colorbar(im, ax=ax, label="Fraction of k-NN", shrink=0.8)

    plt.suptitle("k-NN Overlap: Pixel Space vs VAE Latent Space", fontsize=14)
    plt.tight_layout()
    fig_path = output_dir / "figures" / "knn_overlap_pixel_vs_latent.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    run_latent_overlap()
