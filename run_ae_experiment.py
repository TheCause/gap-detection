#!/usr/bin/env python3
"""
AE Experiment: Deterministic autoencoder vs VAE for gap detection.

Tests two predictions:
  1. Geometry: inter-centroid ratios AE > VAE (KL contraction disappears)
  2. Topology: W(H1) AE > VAE (topological holes better preserved)

Pipeline:
  Phase 1: Train AE for 6 conditions x 3 seeds
  Phase 2: Normalize latents (center + median-distance scaling)
  Phase 3: TDA + Wasserstein comparisons (bootstrap B=100)
  Phase 4: Geometric metrics (inter-centroid ratios, expansion)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ExperimentConfig
from ae import AE
from utils import select_device, load_dataset, train_model, extract_latents, normalize_latents, compute_centroids
from compare import bootstrap_compare, null_distribution


CFG = ExperimentConfig()
DEVICE = select_device()
ABLATIONS = [c for c in CFG.conditions if c != "full"]
AE_DIR = CFG.base_dir / "ae_latents"
VAE_DIR = CFG.latents_dir


# ========== Phase 1: Train AE models ==========
def phase1_train():
    print("=" * 60)
    print("PHASE 1: Training AE models")
    print("=" * 60)
    AE_DIR.mkdir(parents=True, exist_ok=True)
    CFG.models_dir.mkdir(parents=True, exist_ok=True)

    for cond, excluded in CFG.conditions.items():
        for seed in CFG.seeds:
            rname = f"{cond}_seed{seed}"
            latents_file = AE_DIR / f"{rname}_latents.npy"
            if latents_file.exists():
                print(f"  [{rname}] Cached, skipping")
                continue

            print(f"  [{rname}] Training...")
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_ds = load_dataset(excluded_classes=excluded, train=True)
            model = AE(latent_dim=CFG.latent_dim, hidden_dim=CFG.hidden_dim).to(DEVICE)
            model = train_model(model, train_ds, CFG, DEVICE, label=rname)
            torch.save(model.state_dict(), CFG.models_dir / f"ae_{rname}.pt")

            test_ds = load_dataset(excluded_classes=excluded, train=False)
            test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False)
            latents, labels = extract_latents(model, test_loader, DEVICE)
            np.save(latents_file, latents)
            np.save(AE_DIR / f"{rname}_labels.npy", labels)
            print(f"    Latents: {latents.shape}")


# ========== Phase 2: Normalize ==========
def phase2_normalize():
    print("\n" + "=" * 60)
    print("PHASE 2: Normalization")
    print("=" * 60)

    s_values = {}
    for cond in CFG.conditions:
        s_seeds = []
        for seed in CFG.seeds:
            Z = np.load(AE_DIR / f"{cond}_seed{seed}_latents.npy")
            Z_norm, s = normalize_latents(Z)
            np.save(AE_DIR / f"{cond}_seed{seed}_latents_norm.npy", Z_norm)
            s_seeds.append(s)
        s_values[cond] = {"mean": float(np.mean(s_seeds)), "per_seed": s_seeds}
        print(f"  {cond:>15}: s = {s_values[cond]['mean']:.4f}")

    return s_values


# ========== Phase 3: TDA + Bootstrap ==========
def phase3_tda():
    print("\n" + "=" * 60)
    print("PHASE 3: TDA + Bootstrap comparisons")
    print("=" * 60)

    latents = {c: np.load(AE_DIR / f"{c}_seed42_latents_norm.npy") for c in CFG.conditions}
    comparisons = {}
    for cond in ABLATIONS:
        print(f"\n--- full vs {cond} ---")
        comp = bootstrap_compare(latents["full"], latents[cond],
                                 n_bootstrap=CFG.n_bootstrap, n_samples=CFG.tda_n_samples)
        comparisons[f"full_vs_{cond}"] = comp
        w = comp["wasserstein_H1"]
        print(f"  W(H1) = {w['mean']:.2f} +/- {w['std']:.2f} [{w['ci_low']:.2f}, {w['ci_high']:.2f}]")

    print("\n--- Null distribution (AE full split) ---")
    null = null_distribution(latents["full"], n_bootstrap=CFG.n_bootstrap, n_samples=CFG.tda_n_samples)
    print(f"  Null W(H1): mean={null['null_wasserstein_H1']['mean']:.2f}, "
          f"95th={null['null_wasserstein_H1']['ci_high_95']:.2f}")
    return comparisons, null


# ========== Phase 4: Geometric metrics ==========
def phase4_geometry():
    print("\n" + "=" * 60)
    print("PHASE 4: Geometric metrics")
    print("=" * 60)

    geo = {}
    for cond, excluded in CFG.conditions.items():
        surviving = [c for c in range(10) if c not in excluded]
        seed_centroids, seed_radii = [], []
        for seed in CFG.seeds:
            Z = np.load(AE_DIR / f"{cond}_seed{seed}_latents_norm.npy")
            y = np.load(AE_DIR / f"{cond}_seed{seed}_labels.npy").astype(int)
            centroids = compute_centroids(Z, y, classes=surviving)
            radii = {c: float(np.mean(np.linalg.norm(Z[y == c] - centroids[c], axis=1)))
                     for c in centroids}
            seed_centroids.append(centroids)
            seed_radii.append(radii)

        avg_centroids = {c: np.mean([sd[c] for sd in seed_centroids], axis=0) for c in surviving}
        avg_radii = {c: float(np.mean([sd[c] for sd in seed_radii])) for c in surviving}

        classes = sorted(avg_centroids.keys())
        icd = [float(np.linalg.norm(avg_centroids[classes[i]] - avg_centroids[classes[j]]))
               for i in range(len(classes)) for j in range(i + 1, len(classes))]

        geo[cond] = {
            "surviving": surviving, "n_classes": len(surviving),
            "avg_radius": float(np.mean(list(avg_radii.values()))),
            "per_class_radius": {str(c): avg_radii[c] for c in surviving},
            "inter_centroid_mean": float(np.mean(icd)),
            "inter_centroid_std": float(np.std(icd)),
        }
        print(f"  {cond:>15}: avg_radius={geo[cond]['avg_radius']:.4f}, "
              f"ICD_mean={geo[cond]['inter_centroid_mean']:.4f}")
    return geo


# ========== Figures ==========
def save_figures(ae_comparisons, ae_null, ae_geo):
    CFG.figures_dir.mkdir(parents=True, exist_ok=True)
    labels = ["$-7$", "$-3,7$", "$-2,4$", "$-4,9$", "$-3,5,8$"]
    x = np.arange(len(ABLATIONS))
    w = 0.35

    # Try loading VAE data for comparison
    vae_results_path = CFG.base_dir / "results.json"
    gci_results_path = CFG.base_dir / "gci_test_results.json"
    vae_comparisons = {}
    for path in [vae_results_path, gci_results_path]:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            vae_comparisons.update(data.get("comparisons", {}))

    vae_expansion_path = CFG.base_dir / "expansion_test.json"
    vae_exp = json.load(open(vae_expansion_path)) if vae_expansion_path.exists() else {}

    ae_w = [ae_comparisons[f"full_vs_{c}"]["wasserstein_H1"]["mean"] for c in ABLATIONS]
    vae_w = [vae_comparisons.get(f"full_vs_{c}", {}).get("wasserstein_H1", {}).get("mean", 0)
             for c in ABLATIONS]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: W(H1)
    ax = axes[0]
    ax.bar(x - w/2, vae_w, w, label="VAE", color="#1f77b4", alpha=0.8)
    ax.bar(x + w/2, ae_w, w, label="AE", color="#ff7f0e", alpha=0.8)
    ax.axhline(ae_null["null_wasserstein_H1"]["ci_high_95"], color="red", ls="--", alpha=0.5, label="AE null 95th")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("W(H1)"); ax.set_title("Topological signal: AE vs VAE")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Plot 2: ICD ratio
    ax = axes[1]
    ae_full_icd = ae_geo["full"]["inter_centroid_mean"]
    vae_icd_full = vae_exp.get("full", {}).get("inter_centroid", {}).get("mean", 0) if vae_exp else 0
    ae_icd_r = [ae_geo[c]["inter_centroid_mean"] / ae_full_icd for c in ABLATIONS]
    vae_icd_r = [(vae_exp.get(c, {}).get("inter_centroid", {}).get("mean", 0) / vae_icd_full
                  if vae_icd_full else 0) for c in ABLATIONS]
    ax.bar(x - w/2, vae_icd_r, w, label="VAE", color="#1f77b4", alpha=0.8)
    ax.bar(x + w/2, ae_icd_r, w, label="AE", color="#ff7f0e", alpha=0.8)
    ax.axhline(1.0, color="black", ls="--", alpha=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("ICD ratio (ablated / full)"); ax.set_title("Inter-centroid distance ratio")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Expansion ratio
    ax = axes[2]
    ae_full_radii = ae_geo["full"]["per_class_radius"]
    ae_exp_r, vae_exp_r = [], []
    for cond in ABLATIONS:
        surviving = [c for c in range(10) if c not in CFG.conditions[cond]]
        ae_rs = [ae_geo[cond]["per_class_radius"].get(str(c), 0) / ae_full_radii.get(str(c), 1)
                 for c in surviving if ae_full_radii.get(str(c), 0) > 0]
        vae_rs = []
        for c in surviving:
            v_a = vae_exp.get(cond, {}).get("per_class", {}).get(str(c), {}).get("mean_radius", 0)
            v_f = vae_exp.get("full", {}).get("per_class", {}).get(str(c), {}).get("mean_radius", 0)
            if v_f > 0 and v_a > 0:
                vae_rs.append(v_a / v_f)
        ae_exp_r.append(np.mean(ae_rs) if ae_rs else 0)
        vae_exp_r.append(np.mean(vae_rs) if vae_rs else 0)
    ax.bar(x - w/2, vae_exp_r, w, label="VAE", color="#1f77b4", alpha=0.8)
    ax.bar(x + w/2, ae_exp_r, w, label="AE", color="#ff7f0e", alpha=0.8)
    ax.axhline(1.0, color="black", ls="--", alpha=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Expansion ratio"); ax.set_title("Per-class expansion ratio")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = CFG.figures_dir / "ae_vs_vae.png"
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f"Figure saved to {fig_path}")


# ========== Main ==========
def main():
    t0 = time.time()

    phase1_train()
    t1 = time.time()
    print(f"\nPhase 1 done in {t1-t0:.1f}s")

    s_values = phase2_normalize()
    t2 = time.time()
    print(f"Phase 2 done in {t2-t1:.1f}s")

    ae_comparisons, ae_null = phase3_tda()
    t3 = time.time()
    print(f"Phase 3 done in {t3-t2:.1f}s")

    ae_geo = phase4_geometry()
    t4 = time.time()
    print(f"Phase 4 done in {t4-t3:.1f}s")

    # Save results
    output = {
        "model": "AE_deterministic",
        "config": {
            "latent_dim": CFG.latent_dim, "hidden_dim": CFG.hidden_dim,
            "epochs": CFG.epochs, "lr": CFG.lr, "batch_size": CFG.batch_size,
            "seeds": CFG.seeds, "n_bootstrap": CFG.n_bootstrap,
            "normalization": "center_per_dim + median_pairwise_distance_scaling",
        },
        "scale_factors": {c: s_values[c]["mean"] for c in CFG.conditions},
        "comparisons": ae_comparisons,
        "null_distribution": ae_null,
        "geometry": ae_geo,
        "w_h1_summary": {c: ae_comparisons[f"full_vs_{c}"]["wasserstein_H1"]["mean"] for c in ABLATIONS},
        "timing": {
            "training_s": round(t1-t0, 1), "normalization_s": round(t2-t1, 1),
            "tda_bootstrap_s": round(t3-t2, 1), "geometry_s": round(t4-t3, 1),
            "total_s": round(t4-t0, 1),
        },
    }
    out_path = CFG.base_dir / "ae_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    save_figures(ae_comparisons, ae_null, ae_geo)
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
