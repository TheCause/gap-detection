#!/usr/bin/env python3
"""
GCI Validation: Two new ablation conditions to stress-test MNDC vs GCI.

Conditions:
  minus_2_4   {2, 4}  — high MNDC, low cohesion (dispersed isolates)
  minus_4_9   {4, 9}  — lower MNDC, high cohesion (tight pair)

Predictions:
  GCI:  minus_4_9 >> minus_2_4  (cohesion wins)
  MNDC: minus_2_4 >> minus_4_9  (depth wins)

Reuses existing full latents for comparison.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
import time
import numpy as np
from pathlib import Path

from config import ExperimentConfig
from train import train_vae
from tda import compute_persistence, persistence_features, betti_numbers
from compare import bootstrap_compare


def run_gci_test():
    config = ExperimentConfig()
    config.ensure_dirs()

    # New conditions only
    new_conditions = {
        "minus_2_4": [2, 4],
        "minus_4_9": [4, 9],
    }

    # Add to config for train_vae to work
    config.conditions.update(new_conditions)

    from utils import select_device
    device = select_device()
    t0 = time.time()

    # ========== Phase 1: Train new conditions ==========
    print("=" * 60)
    print("Phase 1: Training VAEs for new conditions")
    print("=" * 60)

    latents_by_run = {}
    for condition, excluded in new_conditions.items():
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

    t1 = time.time()
    print(f"\nTraining done in {t1 - t0:.1f}s")

    # Load existing full latents
    for seed in config.seeds:
        run_name = config.run_name("full", seed)
        latents_by_run[("full", seed)] = np.load(
            config.latents_dir / f"{run_name}_latents.npy")

    # ========== Phase 2: TDA per condition ==========
    print("\n" + "=" * 60)
    print("Phase 2: TDA analysis")
    print("=" * 60)

    tda_results = {}
    for condition in new_conditions:
        print(f"\n--- TDA: {condition} ---")
        all_features = []
        for seed in config.seeds:
            latents = latents_by_run[(condition, seed)]
            res = compute_persistence(latents, max_dim=config.tda_max_dim,
                                      n_samples=config.tda_n_samples, seed=seed)
            feat = persistence_features(res["diagrams"])
            bet = betti_numbers(res["diagrams"])
            all_features.append(feat)
            print(f"  seed{seed}: H1={bet['H1']}, "
                  f"H1_total_pers={feat.get('H1_total_persistence', 0):.3f}")

        # Aggregate
        agg = {"excluded": new_conditions[condition]}
        for key in all_features[0]:
            vals = [f[key] for f in all_features if not isinstance(f[key], list)]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals))
        tda_results[condition] = agg

    t2 = time.time()
    print(f"\nTDA done in {t2 - t1:.1f}s")

    # ========== Phase 3: Bootstrap comparisons ==========
    print("\n" + "=" * 60)
    print("Phase 3: Bootstrap comparisons (full vs new conditions)")
    print("=" * 60)

    ref_latents = latents_by_run[("full", config.seeds[0])]
    comparisons = {}

    for condition in new_conditions:
        print(f"\n--- Comparing full vs {condition} ---")
        cond_latents = latents_by_run[(condition, config.seeds[0])]
        comp = bootstrap_compare(ref_latents, cond_latents,
                                 n_bootstrap=config.n_bootstrap,
                                 n_samples=config.tda_n_samples)
        comparisons[f"full_vs_{condition}"] = comp

        w_h1 = comp["wasserstein_H1"]
        print(f"  W_H1: {w_h1['mean']:.2f} +/- {w_h1['std']:.2f} "
              f"[{w_h1['ci_low']:.2f}, {w_h1['ci_high']:.2f}]")

    t3 = time.time()
    print(f"\nBootstrap done in {t3 - t2:.1f}s")

    # ========== Results ==========
    print("\n" + "=" * 60)
    print("RESULTS: GCI STRESS TEST")
    print("=" * 60)

    # Load original results for context
    orig_path = config.base_dir / "results.json"
    if orig_path.exists():
        with open(orig_path) as f:
            orig = json.load(f)
        print("\n--- Original conditions ---")
        for key, val in orig["comparisons"].items():
            w = val["wasserstein_H1"]
            print(f"  {key}: W_H1 = {w['mean']:.2f} [{w['ci_low']:.2f}, {w['ci_high']:.2f}]")

    print("\n--- New conditions ---")
    for key, val in comparisons.items():
        w = val["wasserstein_H1"]
        print(f"  {key}: W_H1 = {w['mean']:.2f} [{w['ci_low']:.2f}, {w['ci_high']:.2f}]")

    # Verdict
    w_24 = comparisons["full_vs_minus_2_4"]["wasserstein_H1"]["mean"]
    w_49 = comparisons["full_vs_minus_4_9"]["wasserstein_H1"]["mean"]

    print(f"\n--- VERDICT ---")
    print(f"  minus_2_4 (high MNDC, low cohesion): W_H1 = {w_24:.2f}")
    print(f"  minus_4_9 (med MNDC, high cohesion):  W_H1 = {w_49:.2f}")

    if w_49 > w_24:
        print(f"  => minus_4_9 > minus_2_4 : COHERENCE WINS -> GCI valide")
    else:
        print(f"  => minus_2_4 > minus_4_9 : PROFONDEUR WINS -> MNDC suffit")

    # Save
    output = {
        "new_conditions": {
            c: {"tda_features": tda_results[c]} for c in new_conditions
        },
        "comparisons": comparisons,
        "predictions": {
            "GCI": "minus_4_9 > minus_2_4",
            "MNDC_only": "minus_2_4 > minus_4_9",
        },
        "observed": {
            "minus_2_4_W_H1": w_24,
            "minus_4_9_W_H1": w_49,
            "winner": "GCI" if w_49 > w_24 else "MNDC",
        },
        "timing": {
            "training_s": round(t1 - t0, 1),
            "tda_s": round(t2 - t1, 1),
            "bootstrap_s": round(t3 - t2, 1),
            "total_s": round(time.time() - t0, 1),
        }
    }

    out_path = config.base_dir / "gci_test_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_gci_test()
