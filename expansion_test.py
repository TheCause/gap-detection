"""
Test the "compensatory expansion" hypothesis:
Do surviving classes spread out more when other classes are removed?

Metrics per class per condition:
1. Mean distance to centroid (radius)
2. Trace of covariance matrix (total variance)
3. Inter-centroid distances (class separation)
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


def class_spread(X, y, classes):
    """Compute per-class spread metrics."""
    results = {}
    for c in classes:
        mask = y == c
        Xc = X[mask]
        if len(Xc) < 2:
            continue
        centroid = Xc.mean(axis=0)
        dists = np.linalg.norm(Xc - centroid, axis=1)
        cov = np.cov(Xc.T)
        results[int(c)] = {
            "mean_radius": float(np.mean(dists)),
            "median_radius": float(np.median(dists)),
            "std_radius": float(np.std(dists)),
            "trace_cov": float(np.trace(cov)),
            "n_points": int(len(Xc)),
            "centroid": centroid.tolist(),
        }
    return results


def inter_centroid_distances(spreads):
    """Compute mean pairwise distance between class centroids."""
    classes = sorted(spreads.keys())
    centroids = np.array([spreads[c]["centroid"] for c in classes])
    n = len(centroids)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    return {
        "mean": float(np.mean(dists)),
        "std": float(np.std(dists)),
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
    }


def run_expansion_test():
    from config import ExperimentConfig
    cfg = ExperimentConfig()
    latents_dir = cfg.latents_dir
    output_dir = cfg.base_dir

    conditions = cfg.conditions
    seeds = cfg.seeds
    # Classes that survive ALL conditions
    all_excluded = set()
    for exc in conditions.values():
        all_excluded.update(exc)
    universal_survivors = [c for c in range(10) if c not in all_excluded]
    print(f"Universal survivors (present in all conditions): {universal_survivors}")
    # That's: {0, 1, 6} — only 3 classes survive everything

    # Better: compare per condition pair
    # For each condition, compute spread of its surviving classes
    all_results = {}

    for cond, excluded in conditions.items():
        surviving = [c for c in range(10) if c not in excluded]
        print(f"\n=== {cond} (excluded: {excluded}, surviving: {surviving}) ===")

        seed_spreads = []
        for seed in seeds:
            fpath = latents_dir / f"{cond}_seed{seed}_latents.npy"
            if not fpath.exists():
                print(f"  SKIP {cond}_seed{seed} (not found)")
                continue
            X = np.load(fpath)
            y = np.load(latents_dir / f"{cond}_seed{seed}_labels.npy").astype(int)

            sp = class_spread(X, y, surviving)
            seed_spreads.append(sp)

        if not seed_spreads:
            continue

        # Average across seeds
        avg_spread = {}
        for c in surviving:
            if c in seed_spreads[0]:
                avg_spread[c] = {
                    "mean_radius": np.mean([s[c]["mean_radius"] for s in seed_spreads]),
                    "trace_cov": np.mean([s[c]["trace_cov"] for s in seed_spreads]),
                    "n_points": seed_spreads[0][c]["n_points"],
                    "centroid": np.mean([s[c]["centroid"] for s in seed_spreads], axis=0).tolist(),
                }

        # Summary stats
        radii = [avg_spread[c]["mean_radius"] for c in surviving if c in avg_spread]
        traces = [avg_spread[c]["trace_cov"] for c in surviving if c in avg_spread]
        icd = inter_centroid_distances(avg_spread)

        all_results[cond] = {
            "excluded": excluded,
            "surviving": surviving,
            "n_classes": len(surviving),
            "per_class": {str(c): {
                "mean_radius": round(avg_spread[c]["mean_radius"], 4),
                "trace_cov": round(avg_spread[c]["trace_cov"], 4),
            } for c in surviving if c in avg_spread},
            "avg_mean_radius": round(float(np.mean(radii)), 4),
            "avg_trace_cov": round(float(np.mean(traces)), 4),
            "total_trace_cov": round(float(np.sum(traces)), 4),
            "inter_centroid": {k: round(v, 4) for k, v in icd.items()},
        }

        print(f"  Avg mean_radius: {np.mean(radii):.4f}")
        print(f"  Avg trace(cov): {np.mean(traces):.4f}")
        print(f"  Total trace(cov): {np.sum(traces):.4f}")
        print(f"  Inter-centroid mean: {icd['mean']:.4f}")

    # ========== Comparison table ==========
    print("\n" + "=" * 70)
    print("EXPANSION TEST: SURVIVING CLASSES SPREAD")
    print("=" * 70)

    # Compare classes that survive in both full and each ablation
    print(f"\n{'Condition':>15} | {'N_cls':>5} | {'Avg radius':>10} | {'Avg trace':>10} | "
          f"{'Total trace':>11} | {'Centroid dist':>13}")
    print("-" * 80)

    for cond in ["full", "minus_7", "minus_3_7", "minus_2_4", "minus_4_9", "minus_cluster"]:
        r = all_results[cond]
        print(f"{cond:>15} | {r['n_classes']:>5} | {r['avg_mean_radius']:>10.4f} | "
              f"{r['avg_trace_cov']:>10.4f} | {r['total_trace_cov']:>11.4f} | "
              f"{r['inter_centroid']['mean']:>13.4f}")

    # Per-class expansion: compare each class in full vs ablations
    print("\n\nPER-CLASS RADIUS COMPARISON (full vs ablations):")
    print(f"{'Class':>6} | {'full':>8} | {'minus_7':>8} | {'minus_3_7':>8} | "
          f"{'minus_2_4':>8} | {'minus_4_9':>8} | {'minus_cl':>8}")
    print("-" * 72)

    for c in range(10):
        row = f"{c:>6} |"
        for cond in ["full", "minus_7", "minus_3_7", "minus_2_4", "minus_4_9", "minus_cluster"]:
            r = all_results[cond]
            if str(c) in r["per_class"]:
                val = r["per_class"][str(c)]["mean_radius"]
                row += f" {val:>8.4f} |"
            else:
                row += f" {'--':>8} |"
        print(row)

    # Compute expansion ratio for common classes
    print("\n\nEXPANSION RATIOS (radius_ablation / radius_full):")
    for cond in ["minus_7", "minus_3_7", "minus_2_4", "minus_4_9", "minus_cluster"]:
        surviving = [c for c in range(10) if c not in conditions[cond]]
        ratios = []
        for c in surviving:
            r_full = all_results["full"]["per_class"].get(str(c), {}).get("mean_radius")
            r_abl = all_results[cond]["per_class"].get(str(c), {}).get("mean_radius")
            if r_full and r_abl:
                ratios.append(r_abl / r_full)
        if ratios:
            print(f"  {cond:>15}: mean ratio = {np.mean(ratios):.4f} "
                  f"(min={np.min(ratios):.4f}, max={np.max(ratios):.4f}), "
                  f"W_H1={['133.48','60.83','116.51','63.12','196.98'][['minus_7','minus_3_7','minus_2_4','minus_4_9','minus_cluster'].index(cond)]}")

    # Save
    with open(output_dir / "expansion_test.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {output_dir / 'expansion_test.json'}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cond_order = ["full", "minus_7", "minus_3_7", "minus_2_4", "minus_4_9", "minus_cluster"]
    cond_labels = ["full\n(10)", "−7\n(9)", "−3,7\n(8)", "−2,4\n(8)", "−4,9\n(8)", "−3,5,8\n(7)"]

    # Plot 1: Avg mean radius
    ax = axes[0]
    vals = [all_results[c]["avg_mean_radius"] for c in cond_order]
    ax.bar(range(len(vals)), vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(cond_labels, fontsize=8)
    ax.set_ylabel("Avg mean radius")
    ax.set_title("Intra-class spread (mean radius)")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: Inter-centroid distance
    ax = axes[1]
    vals = [all_results[c]["inter_centroid"]["mean"] for c in cond_order]
    ax.bar(range(len(vals)), vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(cond_labels, fontsize=8)
    ax.set_ylabel("Mean inter-centroid L2")
    ax.set_title("Class separation (centroid distances)")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Expansion ratio per class for minus_2_4 vs full
    ax = axes[2]
    surviving_24 = [c for c in range(10) if c not in [2, 4]]
    ratios_24 = []
    for c in surviving_24:
        r_f = all_results["full"]["per_class"].get(str(c), {}).get("mean_radius", 0)
        r_a = all_results["minus_2_4"]["per_class"].get(str(c), {}).get("mean_radius", 0)
        ratios_24.append(r_a / r_f if r_f > 0 else 1.0)
    colors = ["#d62728" if r > 1.0 else "#1f77b4" for r in ratios_24]
    ax.bar(range(len(surviving_24)), ratios_24, color=colors, tick_label=surviving_24)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Radius ratio (minus_2_4 / full)")
    ax.set_title("Per-class expansion in minus_2_4")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = output_dir / "figures" / "expansion_test.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    run_expansion_test()
