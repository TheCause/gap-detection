#!/usr/bin/env python3
"""PCA Baseline: topological gap detection without any learned model.

Tests whether corpus gaps are already detectable in a simple linear
projection (PCA to 16D), before any nonlinear encoding.

This establishes the zero-model baseline for the PCA -> AE -> VAE gradient.
Same conditions and TDA pipeline as the main experiments.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")

from config import ExperimentConfig
from utils import load_dataset_flat, compute_centroids, procrustes_align
from compare import bootstrap_compare, null_distribution

ALL_CLASSES = list(range(10))


def run():
    cfg = ExperimentConfig()
    out_dir = Path(__file__).parent / "output" / "pca_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PCA BASELINE: Topological gap detection without learned model")
    print(f"Latent dim: {cfg.latent_dim}, B: {cfg.n_bootstrap}")
    print(f"Conditions: {list(cfg.conditions.keys())}")
    print("=" * 60)

    t_start = time.time()

    # ===== Phase 1: PCA projections =====
    print("\n--- Phase 1: PCA projections ---")
    latents = {}

    for cond, excluded in cfg.conditions.items():
        X_train, _ = load_dataset_flat(cfg.dataset_name, train=True, excluded_classes=excluded)
        pca = PCA(n_components=cfg.latent_dim, random_state=42)
        pca.fit(X_train)

        explained = pca.explained_variance_ratio_.sum()
        print(f"  {cond}: PCA fit on {X_train.shape[0]} train samples, "
              f"explained variance: {explained:.3f}")

        X_test, L_test = load_dataset_flat(cfg.dataset_name, train=False, excluded_classes=excluded)
        Z = pca.transform(X_test)

        np.save(out_dir / f"{cond}_latents.npy", Z)
        np.save(out_dir / f"{cond}_labels.npy", L_test)
        latents[cond] = (Z, L_test)
        print(f"    Projected {Z.shape[0]} test samples -> {Z.shape[1]}D")

    # ===== Phase 2: TDA Bootstrap =====
    print("\n--- Phase 2: TDA Bootstrap ---")
    Z_full = latents["full"][0]

    comparisons = {}
    for cond, excluded in cfg.conditions.items():
        if cond == "full":
            continue
        print(f"\n  {cond} vs full...")
        comp = bootstrap_compare(Z_full, latents[cond][0],
                                 n_samples=cfg.tda_n_samples,
                                 n_bootstrap=cfg.n_bootstrap)
        comparisons[f"full_vs_{cond}"] = comp
        w = comp["wasserstein_H1"]
        print(f"    W(H1) = {w['mean']:.2f} [{w['ci_low']:.2f}, {w['ci_high']:.2f}]")

    print(f"\n  Null distribution...")
    null = null_distribution(Z_full, n_samples=cfg.tda_n_samples,
                             n_bootstrap=cfg.n_bootstrap)
    null_95 = null["null_wasserstein_H1"]["ci_high_95"]
    print(f"    Null 95th = {null_95:.2f}")

    # ===== Phase 3: Geometry (ICD ratios) =====
    print("\n--- Phase 3: Geometry ---")
    full_c = compute_centroids(latents["full"][0], latents["full"][1], ALL_CLASSES)
    geometry = {}

    for cond, excluded in cfg.conditions.items():
        if cond == "full":
            continue
        surviving = [c for c in ALL_CLASSES if c not in excluded]
        Z_c, L_c = latents[cond]
        cond_c = compute_centroids(Z_c, L_c, surviving)

        full_d, abl_d = [], []
        for i, ci in enumerate(surviving):
            for j, cj in enumerate(surviving):
                if i < j:
                    full_d.append(np.linalg.norm(full_c[ci] - full_c[cj]))
                    abl_d.append(np.linalg.norm(cond_c[ci] - cond_c[cj]))
        icd = float(np.mean(abl_d) / np.mean(full_d)) if np.mean(full_d) > 0 else 0

        radii = {}
        for c in surviving:
            pts = Z_c[L_c == c]
            radii[str(c)] = float(np.mean(np.linalg.norm(pts - cond_c[c], axis=1)))

        geometry[cond] = {
            "n_classes": len(surviving),
            "icd_ratio": icd,
            "avg_radius": float(np.mean(list(radii.values()))),
            "inter_centroid_mean": float(np.mean(full_d)),
        }
        print(f"  {cond}: ICD ratio = {icd:.4f}")

    # ===== Phase 4: Ghost centroid aspiration =====
    print("\n--- Phase 4: Ghost centroid aspiration ---")
    all_asp = []
    asp_per_cond = {}

    for cond, excluded in cfg.conditions.items():
        if cond == "full":
            continue
        surviving = [c for c in ALL_CLASSES if c not in excluded]
        abl_c = compute_centroids(*latents[cond], surviving)

        X_full = np.array([full_c[c] for c in surviving])
        X_abl = np.array([abl_c[c] for c in surviving])
        X_aligned, _, _, _ = procrustes_align(X_full, X_abl)

        cond_asp = []
        for gc in excluded:
            ghost = full_c[gc]
            for i, ci in enumerate(surviving):
                d_before = np.linalg.norm(X_full[i] - ghost)
                d_after = np.linalg.norm(X_aligned[i] - ghost)
                alpha = (d_before - d_after) / d_before if d_before > 1e-10 else 0.0
                cond_asp.append({"d_before": float(d_before), "alpha": float(alpha)})

        all_asp.extend(cond_asp)
        alphas_c = [p["alpha"] for p in cond_asp]
        asp_per_cond[cond] = {
            "mean_alpha": float(np.mean(alphas_c)),
            "n_aspirated": int(sum(1 for a in alphas_c if a > 0)),
            "n_total": len(alphas_c),
        }

    d_all = np.array([p["d_before"] for p in all_asp])
    a_all = np.array([p["alpha"] for p in all_asp])
    r_asp, p_asp = pearsonr(d_all, a_all) if len(d_all) > 2 else (0.0, 1.0)

    print(f"  Aspiration: r={r_asp:.4f}, p={p_asp:.4f}, "
          f"mean_alpha={a_all.mean():+.6f}, frac_aspirated={np.mean(a_all > 0):.3f}")

    # ===== Save results =====
    elapsed = time.time() - t_start
    w_h1_summary = {c: comparisons[f"full_vs_{c}"]["wasserstein_H1"]["mean"]
                    for c in cfg.conditions if c != "full"}

    results = {
        "model": "PCA_baseline",
        "config": {
            "latent_dim": cfg.latent_dim,
            "n_bootstrap": cfg.n_bootstrap,
            "n_samples": cfg.tda_n_samples,
            "method": "sklearn PCA, random_state=42, fit on train, project test",
        },
        "comparisons": comparisons,
        "null_distribution": null,
        "geometry": geometry,
        "aspiration": {
            "pearson_r": float(r_asp),
            "pearson_p": float(p_asp),
            "mean_alpha": float(a_all.mean()),
            "frac_aspirated": float(np.mean(a_all > 0)),
            "n_points": len(a_all),
            "per_condition": asp_per_cond,
        },
        "w_h1_summary": w_h1_summary,
        "null_95_h1": float(null_95),
        "timing": {"total_s": elapsed},
    }

    out_path = out_dir / "pca_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("PCA BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Null 95th H1: {null_95:.2f}")
    print(f"\n{'Condition':>15} | {'W(H1)':>8} | {'Signif?':>8} | {'ICD ratio':>9}")
    print("-" * 50)
    for cond in cfg.conditions:
        if cond == "full":
            continue
        w = w_h1_summary[cond]
        sig = "YES" if w > null_95 else "no"
        icd = geometry[cond]["icd_ratio"]
        print(f"{cond:>15} | {w:>8.2f} | {sig:>8} | {icd:>9.4f}")

    print(f"\nAspiration: r={r_asp:.4f}, p={p_asp:.4f}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    run()
