"""
Normalization sanity check: verify that median-distance scaling
preserves W(H1) rankings on existing VAE latents.

Steps:
1. Compute s (median pairwise L2 distance) per condition per seed
2. Normalize latents: Z_norm = (Z - mean) / s
3. Re-compute TDA + Wasserstein on normalized latents
4. Compare rankings with raw (unnormalized) results
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist
import ripser
from persim import wasserstein

def median_pairwise_distance(Z, n_sample=1000, seed=42):
    """Compute median pairwise L2 distance on a subsample."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Z), min(n_sample, len(Z)), replace=False)
    dists = pdist(Z[idx])
    return float(np.median(dists))


def normalize_latents(Z, s=None):
    """Center per-dimension, scale by global scalar s."""
    Z_centered = Z - Z.mean(axis=0)
    if s is None:
        s = median_pairwise_distance(Z_centered)
    Z_norm = Z_centered / s
    return Z_norm, s


def compute_wasserstein_h1(Z1, Z2, n_samples=2000, seed=42):
    """Single-shot Wasserstein H1 between two latent spaces."""
    rng = np.random.RandomState(seed)
    idx1 = rng.choice(len(Z1), min(n_samples, len(Z1)), replace=False)
    idx2 = rng.choice(len(Z2), min(n_samples, len(Z2)), replace=False)

    res1 = ripser.ripser(Z1[idx1], maxdim=1)
    res2 = ripser.ripser(Z2[idx2], maxdim=1)

    dgm1_h1 = res1["dgms"][1]
    dgm2_h1 = res2["dgms"][1]

    w_h1 = wasserstein(dgm1_h1, dgm2_h1)
    return float(w_h1)


def run_check():
    from config import ExperimentConfig
    cfg = ExperimentConfig()
    latents_dir = cfg.latents_dir
    output_dir = cfg.base_dir
    conditions = cfg.conditions
    seeds = cfg.seeds

    # Load W(H1) values from results if available
    raw_w_h1 = {}
    results_path = cfg.base_dir / "results.json"
    if results_path.exists():
        import json
        with open(results_path) as f:
            data = json.load(f)
        for k, v in data.get("comparisons", {}).items():
            cond = k.replace("full_vs_", "")
            raw_w_h1[cond] = v["wasserstein_H1"]["mean"]

    # ========== Step 1: Compute s per condition per seed ==========
    print("=" * 60)
    print("STEP 1: Median pairwise distances (s)")
    print("=" * 60)

    s_values = {}
    for cond in conditions:
        s_seeds = []
        for seed in seeds:
            fpath = latents_dir / f"{cond}_seed{seed}_latents.npy"
            if not fpath.exists():
                continue
            Z = np.load(fpath)
            s = median_pairwise_distance(Z)
            s_seeds.append(s)
        s_mean = np.mean(s_seeds)
        s_std = np.std(s_seeds)
        s_values[cond] = {"mean": s_mean, "std": s_std, "per_seed": s_seeds}
        print(f"  {cond:>15}: s = {s_mean:.4f} +/- {s_std:.4f}  {s_seeds}")

    # Check: are s values similar across conditions?
    all_s = [s_values[c]["mean"] for c in conditions]
    print(f"\n  Range of s: [{min(all_s):.4f}, {max(all_s):.4f}]")
    print(f"  CV of s: {np.std(all_s)/np.mean(all_s):.4f}")

    # ========== Step 2: Single-shot W(H1) raw vs normalized ==========
    print("\n" + "=" * 60)
    print("STEP 2: W(H1) raw vs normalized (single-shot, seed 42)")
    print("=" * 60)

    # Load seed 42 for all conditions
    latents = {}
    for cond in conditions:
        fpath = latents_dir / f"{cond}_seed42_latents.npy"
        if fpath.exists():
            latents[cond] = np.load(fpath)

    # Raw W(H1): full vs each ablation
    print("\n--- Raw (no normalization) ---")
    raw_results = {}
    for cond in ["minus_7", "minus_3_7", "minus_2_4", "minus_4_9", "minus_cluster"]:
        w = compute_wasserstein_h1(latents["full"], latents[cond])
        raw_results[cond] = w
        print(f"  full vs {cond:>15}: W(H1) = {w:.2f}")

    # Normalized W(H1)
    print("\n--- Normalized (center + median-distance scaling) ---")
    norm_latents = {}
    for cond in conditions:
        Z_norm, s = normalize_latents(latents[cond])
        norm_latents[cond] = Z_norm
        print(f"  {cond:>15}: s = {s:.4f}, "
              f"norm range = [{Z_norm.min():.3f}, {Z_norm.max():.3f}]")

    norm_results = {}
    for cond in ["minus_7", "minus_3_7", "minus_2_4", "minus_4_9", "minus_cluster"]:
        w = compute_wasserstein_h1(norm_latents["full"], norm_latents[cond])
        norm_results[cond] = w
        print(f"  full vs {cond:>15}: W(H1) = {w:.2f}")

    # ========== Step 3: Compare rankings ==========
    print("\n" + "=" * 60)
    print("STEP 3: Ranking comparison")
    print("=" * 60)

    conds = ["minus_7", "minus_3_7", "minus_2_4", "minus_4_9", "minus_cluster"]

    raw_ranking = sorted(conds, key=lambda c: raw_results[c], reverse=True)
    norm_ranking = sorted(conds, key=lambda c: norm_results[c], reverse=True)
    bootstrap_ranking = sorted(conds, key=lambda c: raw_w_h1[c], reverse=True)

    print(f"\n{'':>20} | {'Bootstrap (B=100)':>20} | {'Raw (seed42)':>20} | {'Normalized':>20}")
    print("-" * 85)
    for i in range(5):
        b = f"{bootstrap_ranking[i]} ({raw_w_h1[bootstrap_ranking[i]]:.1f})"
        r = f"{raw_ranking[i]} ({raw_results[raw_ranking[i]]:.1f})"
        n = f"{norm_ranking[i]} ({norm_results[norm_ranking[i]]:.2f})"
        print(f"  Rank {i+1:>12} | {b:>20} | {r:>20} | {n:>20}")

    ranking_preserved = (raw_ranking == norm_ranking)
    bootstrap_match = (bootstrap_ranking == norm_ranking)
    print(f"\n  Raw vs Normalized ranking preserved: {ranking_preserved}")
    print(f"  Bootstrap vs Normalized ranking match: {bootstrap_match}")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    raw_vals = [raw_results[c] for c in conds]
    norm_vals = [norm_results[c] for c in conds]
    boot_vals = [raw_w_h1[c] for c in conds]

    r_raw_norm, _ = spearmanr(raw_vals, norm_vals)
    r_boot_norm, _ = spearmanr(boot_vals, norm_vals)
    print(f"  Spearman(raw, norm): {r_raw_norm:.4f}")
    print(f"  Spearman(bootstrap, norm): {r_boot_norm:.4f}")

    # ========== Step 4: Scale factor analysis ==========
    print("\n" + "=" * 60)
    print("STEP 4: How normalization changes absolute values")
    print("=" * 60)

    print(f"\n{'Condition':>15} | {'Raw W(H1)':>10} | {'Norm W(H1)':>10} | "
          f"{'Ratio':>8} | {'s_full':>8} | {'s_abl':>8}")
    print("-" * 70)

    s_full = median_pairwise_distance(latents["full"] - latents["full"].mean(axis=0))
    for cond in conds:
        Z_abl = latents[cond]
        s_abl = median_pairwise_distance(Z_abl - Z_abl.mean(axis=0))
        ratio = norm_results[cond] / raw_results[cond] if raw_results[cond] > 0 else 0
        print(f"{cond:>15} | {raw_results[cond]:>10.2f} | {norm_results[cond]:>10.2f} | "
              f"{ratio:>8.4f} | {s_full:>8.4f} | {s_abl:>8.4f}")

    # Save
    output = {
        "s_values": {c: s_values[c]["mean"] for c in conditions},
        "s_cv": float(np.std(all_s) / np.mean(all_s)),
        "raw_single_shot": raw_results,
        "normalized_single_shot": norm_results,
        "bootstrap_reference": raw_w_h1,
        "ranking_preserved": ranking_preserved,
        "bootstrap_ranking_match": bootstrap_match,
        "spearman_raw_norm": float(r_raw_norm),
        "spearman_boot_norm": float(r_boot_norm),
    }
    out_path = output_dir / "normalization_check.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_check()
