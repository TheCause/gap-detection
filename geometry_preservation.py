"""
Geometry preservation test: do ablated models conserve the relative
arrangement of surviving classes, or reorganize entirely?

For each ablation, compare the inter-centroid distance matrix of
surviving classes to the corresponding submatrix from the full model.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from utils import compute_centroids


def centroid_distance_matrix(centroids, classes):
    """Pairwise L2 distance matrix between centroids."""
    n = len(classes)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(centroids[classes[i]] - centroids[classes[j]])
    return D


def upper_triangle(D):
    """Extract upper triangle values (excluding diagonal)."""
    n = D.shape[0]
    idx = np.triu_indices(n, k=1)
    return D[idx]


def run_geometry_test():
    from config import ExperimentConfig
    cfg = ExperimentConfig()
    latents_dir = cfg.latents_dir
    output_dir = cfg.base_dir

    conditions = cfg.conditions
    seeds = cfg.seeds

    # Load W(H1) values from results if available, else use defaults
    results_path = cfg.base_dir / "results.json"
    w_h1_map = {}
    if results_path.exists():
        import json
        with open(results_path) as f:
            data = json.load(f)
        for k, v in data.get("comparisons", {}).items():
            cond = k.replace("full_vs_", "")
            w_h1_map[cond] = v["wasserstein_H1"]["mean"]

    # Compute centroids for all conditions, averaged across seeds
    all_centroids = {}
    for cond, excluded in conditions.items():
        surviving = [c for c in range(10) if c not in excluded]
        seed_centroids = []
        for seed in seeds:
            fpath = latents_dir / f"{cond}_seed{seed}_latents.npy"
            if not fpath.exists():
                continue
            X = np.load(fpath)
            y = np.load(latents_dir / f"{cond}_seed{seed}_labels.npy").astype(int)
            seed_centroids.append(compute_centroids(X, y, surviving))

        # Average centroids across seeds
        avg = {}
        for c in surviving:
            avg[c] = np.mean([sc[c] for sc in seed_centroids], axis=0)
        all_centroids[cond] = avg

    # Full model distance matrix (all 10 classes)
    full_classes = list(range(10))
    D_full = centroid_distance_matrix(all_centroids["full"], full_classes)

    print("=" * 70)
    print("GEOMETRY PRESERVATION TEST")
    print("=" * 70)

    results = {}
    for cond, excluded in conditions.items():
        if cond == "full":
            continue

        surviving = sorted([c for c in range(10) if c not in excluded])

        # Distance matrix from ablated model
        D_abl = centroid_distance_matrix(all_centroids[cond], surviving)

        # Corresponding submatrix from full model
        idx = [full_classes.index(c) for c in surviving]
        D_full_sub = D_full[np.ix_(idx, idx)]

        # Extract upper triangles for comparison
        ut_full = upper_triangle(D_full_sub)
        ut_abl = upper_triangle(D_abl)

        # Metric 1: Pearson correlation of distances
        r_pearson, p_pearson = pearsonr(ut_full, ut_abl)

        # Metric 2: Spearman rank correlation (ordering preserved?)
        r_spearman, p_spearman = spearmanr(ut_full, ut_abl)

        # Metric 3: Distance ratios (ablated / full)
        ratios = ut_abl / ut_full
        ratio_mean = np.mean(ratios)
        ratio_std = np.std(ratios)
        ratio_cv = ratio_std / ratio_mean  # coefficient of variation

        # Metric 4: Procrustes-like residual (normalize both to unit scale, then compare)
        ut_full_norm = ut_full / np.mean(ut_full)
        ut_abl_norm = ut_abl / np.mean(ut_abl)
        residual = np.sqrt(np.mean((ut_full_norm - ut_abl_norm) ** 2))

        results[cond] = {
            "excluded": excluded,
            "n_surviving": len(surviving),
            "n_pairs": len(ut_full),
            "pearson_r": round(r_pearson, 4),
            "spearman_r": round(r_spearman, 4),
            "ratio_mean": round(ratio_mean, 4),
            "ratio_std": round(ratio_std, 4),
            "ratio_cv": round(ratio_cv, 4),
            "residual_normalized": round(residual, 4),
            "w_h1": w_h1_map[cond],
        }

        print(f"\n--- {cond} (removed: {excluded}) ---")
        print(f"  Pearson r  = {r_pearson:.4f} (p={p_pearson:.2e})")
        print(f"  Spearman r = {r_spearman:.4f} (p={p_spearman:.2e})")
        print(f"  Ratio mean = {ratio_mean:.4f}, std = {ratio_std:.4f}, CV = {ratio_cv:.4f}")
        print(f"  Normalized residual = {residual:.4f}")
        print(f"  W_H1 = {w_h1_map[cond]}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Geometry preservation vs Topological signal")
    print("=" * 70)
    print(f"{'Condition':>15} | {'Pearson':>8} | {'Spearman':>8} | "
          f"{'Ratio CV':>8} | {'Residual':>8} | {'W_H1':>8}")
    print("-" * 70)
    for cond in ["minus_7", "minus_3_7", "minus_2_4", "minus_4_9", "minus_cluster"]:
        r = results[cond]
        print(f"{cond:>15} | {r['pearson_r']:>8.4f} | {r['spearman_r']:>8.4f} | "
              f"{r['ratio_cv']:>8.4f} | {r['residual_normalized']:>8.4f} | "
              f"{r['w_h1']:>8.2f}")

    # Test: does geometry preservation correlate with W_H1?
    conds = ["minus_7", "minus_3_7", "minus_2_4", "minus_4_9", "minus_cluster"]
    residuals = [results[c]["residual_normalized"] for c in conds]
    w_h1s = [results[c]["w_h1"] for c in conds]
    r_res_wh1, _ = pearsonr(residuals, w_h1s)
    print(f"\nCorrelation(residual, W_H1) = {r_res_wh1:.4f}")

    cvs = [results[c]["ratio_cv"] for c in conds]
    r_cv_wh1, _ = pearsonr(cvs, w_h1s)
    print(f"Correlation(ratio_CV, W_H1) = {r_cv_wh1:.4f}")

    pearsons = [results[c]["pearson_r"] for c in conds]
    r_pear_wh1, _ = pearsonr(pearsons, w_h1s)
    print(f"Correlation(pearson_r, W_H1) = {r_pear_wh1:.4f}")
    print(f"  (negative = less preservation → more W_H1 signal)")

    # Save
    with open(output_dir / "geometry_preservation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir / 'geometry_preservation.json'}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    labels = ["−7", "−3,7", "−2,4", "−4,9", "−3,5,8"]
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Plot 1: Ratio distributions
    ax = axes[0]
    for i, cond in enumerate(conds):
        surviving = sorted([c for c in range(10) if c not in conditions[cond]])
        idx = [full_classes.index(c) for c in surviving]
        D_abl = centroid_distance_matrix(all_centroids[cond], surviving)
        D_full_sub = D_full[np.ix_(idx, idx)]
        ratios = upper_triangle(D_abl) / upper_triangle(D_full_sub)
        ax.boxplot(ratios, positions=[i], widths=0.6,
                   boxprops=dict(color=colors[i]),
                   medianprops=dict(color=colors[i]))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Distance ratio (ablated / full)")
    ax.set_title("Distance ratio distributions")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: Normalized residual vs W_H1
    ax = axes[1]
    for i, cond in enumerate(conds):
        ax.scatter(results[cond]["residual_normalized"], results[cond]["w_h1"],
                   s=100, color=colors[i], zorder=5)
        ax.annotate(labels[i],
                    (results[cond]["residual_normalized"], results[cond]["w_h1"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Normalized residual (geometry change)")
    ax.set_ylabel("W_H1 (topological signal)")
    ax.set_title(f"Geometry change vs Signal (r={r_res_wh1:.3f})")
    ax.grid(True, alpha=0.3)

    # Plot 3: Pearson correlation vs W_H1
    ax = axes[2]
    for i, cond in enumerate(conds):
        ax.scatter(results[cond]["pearson_r"], results[cond]["w_h1"],
                   s=100, color=colors[i], zorder=5)
        ax.annotate(labels[i],
                    (results[cond]["pearson_r"], results[cond]["w_h1"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Pearson r (geometry preservation)")
    ax.set_ylabel("W_H1 (topological signal)")
    ax.set_title(f"Preservation vs Signal (r={r_pear_wh1:.3f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "figures" / "geometry_preservation.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    run_geometry_test()
