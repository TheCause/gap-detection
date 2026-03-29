"""
MNDC (Mean Nearest-Neighbor Distance to Complement)
Predictive metric for topological gap signal strength.
Computes on full latent space without needing ablation.
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


def mndc(X_s, X_r):
    """
    Mean Nearest-Neighbor Distance to Complement.
    For each point in X_s, find distance to nearest point in X_r.
    Return mean, median, std, and the full distribution.
    """
    # Batched computation to avoid memory issues
    X_r_sq = np.sum(X_r ** 2, axis=1)  # (N_r,)
    batch_size = 2000
    all_min_dists = []

    for start in range(0, len(X_s), batch_size):
        end = min(start + batch_size, len(X_s))
        batch = X_s[start:end]
        batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)  # (B, 1)
        dists_sq = batch_sq + X_r_sq[np.newaxis, :] - 2.0 * batch @ X_r.T
        np.maximum(dists_sq, 0, out=dists_sq)  # numerical safety
        min_dists = np.sqrt(np.min(dists_sq, axis=1))
        all_min_dists.append(min_dists)

    all_min_dists = np.concatenate(all_min_dists)
    return {
        "mean": float(np.mean(all_min_dists)),
        "median": float(np.median(all_min_dists)),
        "std": float(np.std(all_min_dists)),
        "p25": float(np.percentile(all_min_dists, 25)),
        "p75": float(np.percentile(all_min_dists, 75)),
        "n_points": len(all_min_dists),
    }


def run_mndc_analysis():
    from config import ExperimentConfig
    cfg = ExperimentConfig()
    latents_dir = cfg.latents_dir
    output_dir = cfg.base_dir
    seeds = cfg.seeds

    # Use all ablation conditions (not just 3)
    conditions = {k: v for k, v in cfg.conditions.items() if k != "full"}

    # Load W(H1) values from results if available
    w_h1 = {}
    results_path = cfg.base_dir / "results.json"
    if results_path.exists():
        import json
        with open(results_path) as f:
            data = json.load(f)
        for k, v in data.get("comparisons", {}).items():
            cond = k.replace("full_vs_", "")
            w_h1[cond] = v["wasserstein_H1"]["mean"]

    results = {}
    for cond, excluded in conditions.items():
        seed_mndcs = []
        for seed in seeds:
            X = np.load(latents_dir / f"full_seed{seed}_latents.npy")
            y = np.load(latents_dir / f"full_seed{seed}_labels.npy").astype(int)

            mask_s = np.isin(y, excluded)
            mask_r = ~mask_s
            X_s = X[mask_s]
            X_r = X[mask_r]

            m = mndc(X_s, X_r)
            seed_mndcs.append(m)
            print(f"  {cond} seed {seed}: MNDC mean={m['mean']:.4f}, "
                  f"median={m['median']:.4f}, n={m['n_points']}")

        # Average across seeds
        avg_mean = np.mean([m["mean"] for m in seed_mndcs])
        avg_median = np.mean([m["median"] for m in seed_mndcs])
        avg_std = np.mean([m["std"] for m in seed_mndcs])

        results[cond] = {
            "excluded": excluded,
            "mndc_mean": round(avg_mean, 4),
            "mndc_median": round(avg_median, 4),
            "mndc_std": round(avg_std, 4),
            "w_h1": w_h1[cond],
            "per_seed": seed_mndcs,
        }
        print(f"\n  {cond} AVG: MNDC={avg_mean:.4f}, W_H1={w_h1[cond]}")

    # Also compute MNDC for all 10 individual classes
    print("\n=== MNDC PER INDIVIDUAL CLASS (seed 42) ===")
    X = np.load(latents_dir / "full_seed42_latents.npy")
    y = np.load(latents_dir / "full_seed42_labels.npy").astype(int)

    class_mndcs = {}
    for c in range(10):
        mask_s = y == c
        mask_r = ~mask_s
        m = mndc(X[mask_s], X[mask_r])
        class_mndcs[c] = m["mean"]
        print(f"  Class {c}: MNDC={m['mean']:.4f} (n={m['n_points']})")

    # Correlation analysis
    print("\n=== CORRELATION: MNDC vs W_H1 ===")
    conds = ["minus_3_7", "minus_7", "minus_cluster"]
    mndc_vals = [results[c]["mndc_mean"] for c in conds]
    wh1_vals = [results[c]["w_h1"] for c in conds]

    for c in conds:
        print(f"  {c:>15}: MNDC={results[c]['mndc_mean']:.4f}, W_H1={results[c]['w_h1']}")

    # Pearson correlation (3 points, indicative only)
    if len(set(mndc_vals)) > 1:
        r = np.corrcoef(mndc_vals, wh1_vals)[0, 1]
        print(f"\n  Pearson r = {r:.4f}")

    # Linear fit
    coeffs = np.polyfit(mndc_vals, wh1_vals, 1)
    print(f"  Linear fit: W_H1 = {coeffs[0]:.2f} * MNDC + {coeffs[1]:.2f}")

    # Save
    save_data = {
        "conditions": {c: {
            "excluded": results[c]["excluded"],
            "mndc_mean": results[c]["mndc_mean"],
            "mndc_median": results[c]["mndc_median"],
            "w_h1": results[c]["w_h1"],
        } for c in conds},
        "individual_classes": {str(c): round(v, 4) for c, v in class_mndcs.items()},
        "correlation": {
            "pearson_r": round(float(r), 4) if len(set(mndc_vals)) > 1 else None,
            "linear_slope": round(float(coeffs[0]), 2),
            "linear_intercept": round(float(coeffs[1]), 2),
        }
    }
    with open(output_dir / "mndc_analysis.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {output_dir / 'mndc_analysis.json'}")

    # Plot: MNDC vs W_H1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MNDC vs W_H1 scatter
    ax = axes[0]
    for c in conds:
        ax.scatter(results[c]["mndc_mean"], results[c]["w_h1"],
                   s=100, zorder=5)
        ax.annotate(c, (results[c]["mndc_mean"], results[c]["w_h1"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    # Fit line
    x_fit = np.linspace(min(mndc_vals) * 0.9, max(mndc_vals) * 1.1, 50)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, "--", color="gray", alpha=0.7,
            label=f"r = {r:.3f}" if len(set(mndc_vals)) > 1 else "")
    ax.set_xlabel("MNDC (mean nearest distance to complement)")
    ax.set_ylabel("Wasserstein H1")
    ax.set_title("MNDC vs Topological Signal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: MNDC per individual class
    ax = axes[1]
    classes = sorted(class_mndcs.keys())
    vals = [class_mndcs[c] for c in classes]
    colors = ["#d62728" if c in [3, 5, 7, 8] else "#1f77b4" for c in classes]
    ax.bar(classes, vals, color=colors)
    ax.set_xlabel("Digit class")
    ax.set_ylabel("MNDC")
    ax.set_title("MNDC per class (red = used in ablations)")
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = output_dir / "figures" / "mndc_analysis.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    run_mndc_analysis()
