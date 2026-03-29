#!/usr/bin/env python3
"""Orchestrator: train all conditions -> TDA -> compare -> report."""

import json
import sys
import time
import pickle
import numpy as np
from pathlib import Path

from config import ExperimentConfig
from train import train_vae
from tda import compute_persistence, betti_numbers, persistence_features, compute_wasserstein
from compare import bootstrap_compare, null_distribution
from visualize import (
    plot_persistence_diagrams, plot_umap_latents,
    plot_betti_comparison, plot_wasserstein_matrix, generate_report
)

from utils import select_device


def run(config=None):
    if config is None:
        config = ExperimentConfig()
    config.ensure_dirs()
    device = select_device()
    print(f"=== Gap Detection Experiment ===")
    print(f"Device: {device}")
    print(f"Conditions: {list(config.conditions.keys())}")
    print(f"Seeds: {config.seeds}")
    print()

    t0 = time.time()

    # ========== Phase 1: Training ==========
    print("== Phase 1: Training VAEs ==")
    all_latents = {}   # condition -> latents (from first seed, for viz)
    all_labels = {}    # condition -> labels
    latents_by_run = {}  # (condition, seed) -> latents

    for condition, excluded in config.conditions.items():
        print(f"\n--- Condition: {condition} (excluded: {excluded}) ---")
        for seed in config.seeds:
            run_name = config.run_name(condition, seed)
            latents_file = config.latents_dir / f"{run_name}_latents.npy"

            if latents_file.exists():
                print(f"  [{run_name}] Loading cached latents")
                latents = np.load(latents_file)
                labels = np.load(config.latents_dir / f"{run_name}_labels.npy")
            else:
                _, latents, labels = train_vae(config, condition, seed, device=device)

            latents_by_run[(condition, seed)] = latents
            if seed == config.seeds[0]:
                all_latents[condition] = latents
                all_labels[condition] = labels

    t1 = time.time()
    print(f"\nTraining done in {t1-t0:.0f}s")

    # ========== Phase 2: TDA per condition ==========
    print("\n== Phase 2: TDA Analysis ==")
    results = {}

    for condition in config.conditions:
        print(f"\n--- TDA: {condition} ---")
        all_features = []
        all_betti = []
        first_diagrams = None

        for seed in config.seeds:
            latents = latents_by_run[(condition, seed)]
            res = compute_persistence(latents, max_dim=config.tda_max_dim,
                                      n_samples=config.tda_n_samples, seed=seed)
            if first_diagrams is None:
                first_diagrams = res["diagrams"]

            feat = persistence_features(res["diagrams"])
            bet = betti_numbers(res["diagrams"])
            all_features.append(feat)
            all_betti.append(bet)
            print(f"  seed{seed}: H0={bet['H0']}, H1={bet['H1']}, "
                  f"H1_total_pers={feat.get('H1_total_persistence', 0):.3f}")

        # Aggregate across seeds
        agg = {"excluded": config.conditions[condition]}
        for key in all_features[0]:
            vals = [f[key] for f in all_features if not isinstance(f[key], list)]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))

        results[condition] = {"tda_features": agg, "diagrams": first_diagrams}

        # Save diagrams
        with open(config.diagrams_dir / f"{condition}_diagrams.pkl", "wb") as f:
            pickle.dump(first_diagrams, f)

    t2 = time.time()
    print(f"\nTDA done in {t2-t1:.0f}s")

    # ========== Phase 3: Pairwise Comparisons ==========
    print("\n== Phase 3: Bootstrap Comparisons ==")
    comparisons = {}

    # Compare each lacunary vs full (seed[0] for main bootstrap)
    ref_latents = latents_by_run[("full", config.seeds[0])]
    for condition in config.conditions:
        if condition == "full":
            continue
        print(f"\n--- Comparing full vs {condition} ---")
        cond_latents = latents_by_run[(condition, config.seeds[0])]
        comp = bootstrap_compare(ref_latents, cond_latents,
                                 n_bootstrap=config.n_bootstrap,
                                 n_samples=config.tda_n_samples)
        comparisons[f"full_vs_{condition}"] = comp

    # Per-seed W(H1) for "Seeds > null" count
    print("\n--- Per-seed Wasserstein (all seeds) ---")
    per_seed_w = {}
    for condition in config.conditions:
        if condition == "full":
            continue
        seed_vals = []
        for seed in config.seeds:
            ref = latents_by_run[("full", seed)]
            abl = latents_by_run[(condition, seed)]
            p_ref = compute_persistence(ref, max_dim=config.tda_max_dim,
                                        n_samples=config.tda_n_samples, seed=seed)
            p_abl = compute_persistence(abl, max_dim=config.tda_max_dim,
                                        n_samples=config.tda_n_samples, seed=seed)
            w = compute_wasserstein(p_ref["diagrams"], p_abl["diagrams"])
            seed_vals.append({"seed": seed, "W_H1": w["H1"], "W_H0": w["H0"]})
        per_seed_w[condition] = seed_vals
        vals = [s["W_H1"] for s in seed_vals]
        print(f"  {condition}: W(H1) per seed = {[f'{v:.1f}' for v in vals]}")

    # Null distribution
    print("\n--- Null distribution (full vs self) ---")
    null_stats = null_distribution(ref_latents, n_bootstrap=config.n_bootstrap,
                                   n_samples=config.tda_n_samples)

    t3 = time.time()
    print(f"\nComparisons done in {t3-t2:.0f}s")

    # ========== Checkpoint: save results BEFORE visualization ==========
    print("\n== Saving results (checkpoint) ==")

    json_results = {}
    for c, data in results.items():
        json_results[c] = {"tda_features": data["tda_features"]}

    output = {
        "results": json_results,
        "comparisons": comparisons,
        "per_seed_wasserstein": per_seed_w,
        "null_distribution": null_stats,
        "config": {
            "latent_dim": config.latent_dim,
            "epochs": config.epochs,
            "seeds": config.seeds,
            "n_bootstrap": config.n_bootstrap,
            "tda_n_samples": config.tda_n_samples,
        },
        "timing": {
            "training_s": t1 - t0,
            "tda_s": t2 - t1,
            "comparison_s": t3 - t2,
            "total_s": time.time() - t0,
        }
    }

    with open(config.base_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {config.base_dir / 'results.json'}")

    # Save report immediately
    report_results = {c: {"tda_features": results[c]["tda_features"],
                          "excluded": config.conditions[c]}
                      for c in results}
    generate_report(report_results, comparisons, null_stats, config.base_dir)

    # ========== Phase 4: Visualization (each step protected) ==========
    print("\n== Phase 4: Visualization ==")

    diag_dict = {c: results[c]["diagrams"] for c in results}
    try:
        plot_persistence_diagrams(diag_dict, config.figures_dir)
    except Exception as e:
        print(f"  WARN: persistence_diagrams failed: {e}")

    try:
        plot_umap_latents(all_latents, all_labels, config.figures_dir)
    except Exception as e:
        print(f"  WARN: umap_latents failed: {e}")

    betti_results = {c: {"tda_features": results[c]["tda_features"]} for c in results}
    try:
        plot_betti_comparison(betti_results, config.figures_dir)
    except Exception as e:
        print(f"  WARN: betti_comparison failed: {e}")

    try:
        plot_wasserstein_matrix(comparisons, config.figures_dir)
    except Exception as e:
        print(f"  WARN: wasserstein_matrix failed: {e}")

    total = time.time() - t0
    print(f"\n=== DONE in {total:.0f}s ===")

    # Quick summary
    print("\n--- Quick Summary ---")
    for key, data in comparisons.items():
        wh1 = data["wasserstein_H1"]
        sig = "SIGNIFICANT" if wh1["ci_low"] > 0 else "not significant"
        print(f"  {key}: W_H1={wh1['mean']:.3f} [{wh1['ci_low']:.3f}, {wh1['ci_high']:.3f}] — {sig}")

    nh1 = null_stats["null_wasserstein_H1"]
    print(f"  Null H1: mean={nh1['mean']:.3f}, 95th={nh1['ci_high_95']:.3f}")


if __name__ == "__main__":
    run()
