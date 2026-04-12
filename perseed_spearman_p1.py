#!/usr/bin/env python3
"""Per-seed Spearman analysis for P1 (aspiration monotonicity).

Runs on latent artifacts produced under the V1 protocol.
Set artifact_status explicitly to distinguish original-artifact
reanalysis from rerun-based replication.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import numpy as np
from scipy.stats import spearmanr, pearsonr

# === PATHS (V1 artifacts on M4) ===
BASE = "/Users/regis/dev/epistemologue/output/experiment"
VAE_DIR = os.path.join(BASE, "latents")
AE_DIR = os.path.join(BASE, "ae_latents")

# === V1 CONFIG ===
V1_PUBLISHED_SEEDS = {42, 123, 456}
EXPECTED_POINTS_PER_SEED = 78  # 9 + 16 + 16 + 16 + 21

CONDITIONS = {
    "minus_7": [7],
    "minus_3_7": [3, 7],
    "minus_2_4": [2, 4],
    "minus_4_9": [4, 9],
    "minus_cluster": [3, 5, 8],
}
ALL_CLASSES = list(range(10))


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact: {path}")
    return path


def detect_seeds(latents_dir):
    seeds = set()
    for f in os.listdir(latents_dir):
        if f.startswith("full_seed") and f.endswith("_latents.npy"):
            s = int(f.replace("full_seed", "").replace("_latents.npy", ""))
            seeds.add(s)
    return sorted(seeds)


def compute_centroids(Z, labels, classes=None):
    """Per-class centroids — IDENTICAL to V1 utils.py:93-97."""
    if classes is None:
        classes = sorted(set(int(x) for x in labels))
    return {int(c): Z[labels == c].mean(axis=0) for c in classes if (labels == c).sum() > 0}


def procrustes_align(X_ref, X_tgt):
    """Procrustes alignment — IDENTICAL to V1 utils.py:100-118.
    Returns: (X_aligned, R, s, t).
    """
    mu_r = X_ref.mean(axis=0)
    mu_t = X_tgt.mean(axis=0)
    Xr = X_ref - mu_r
    Xt = X_tgt - mu_t
    U, S, Vt = np.linalg.svd(Xr.T @ Xt)
    d = np.linalg.det(U @ Vt)
    D = np.eye(len(S))
    D[-1, -1] = np.sign(d)
    R = Vt.T @ D @ U.T
    Xt_rot = Xt @ R
    s = np.trace(Xr.T @ Xt_rot) / np.trace(Xt_rot.T @ Xt_rot)
    t = mu_r - s * (mu_t @ R)
    X_aligned = s * X_tgt @ R + t
    return X_aligned, R, s, t


def aspiration_for_seed(latents_dir, seed, cond, excluded):
    surviving = [c for c in ALL_CLASSES if c not in excluded]

    Z_full = np.load(require(os.path.join(latents_dir, f"full_seed{seed}_latents.npy")))
    L_full = np.load(require(os.path.join(latents_dir, f"full_seed{seed}_labels.npy")))
    full_c = compute_centroids(Z_full, L_full, classes=ALL_CLASSES)

    Z_abl = np.load(require(os.path.join(latents_dir, f"{cond}_seed{seed}_latents.npy")))
    L_abl = np.load(require(os.path.join(latents_dir, f"{cond}_seed{seed}_labels.npy")))
    abl_c = compute_centroids(Z_abl, L_abl, classes=surviving)

    X_full = np.array([full_c[c] for c in surviving])
    X_abl = np.array([abl_c[c] for c in surviving])
    # procrustes_align(X_ref, X_tgt) — align X_tgt (ablated) to X_ref (full)
    X_aligned, _, _, _ = procrustes_align(X_full, X_abl)

    points = []
    for gc in excluded:
        ghost = full_c[gc]
        for i, ci in enumerate(surviving):
            d_before = float(np.linalg.norm(X_full[i] - ghost))
            d_after = float(np.linalg.norm(X_aligned[i] - ghost))
            alpha = (d_before - d_after) / d_before if d_before > 1e-10 else 0.0
            points.append({"d_before": d_before, "alpha": alpha})

    return points


def bootstrap_spearman_point_level(x, y, n=1000, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(x), len(x), replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            vals.append(r)
    return np.mean(vals), np.std(vals)


def compute_verdict(rhos, label, n_seeds_for_pass):
    n_neg = sum(1 for r in rhos if r < 0)
    std = np.std(rhos) if len(rhos) > 1 else 0.0
    if n_neg >= n_seeds_for_pass and std < 0.15:
        verdict = "PASS"
    elif n_neg >= max(n_seeds_for_pass - 1, 1):
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    print(f"  {label}: {verdict} ({n_neg}/{len(rhos)} negative, std={std:.4f})")
    return verdict, n_neg


def compute_global(d_list, a_list):
    d_arr = np.array(d_list)
    a_arr = np.array(a_list)
    if len(d_arr) > 2:
        rho, p = spearmanr(d_arr, a_arr)
        bs_mean, bs_std = bootstrap_spearman_point_level(d_arr, a_arr, n=1000)
    else:
        rho, p, bs_mean, bs_std = 0.0, 1.0, 0.0, 0.0
    return {
        "rho": float(rho), "p": float(p), "n_points": len(d_arr),
        "bootstrap_point_mean": float(bs_mean),
        "bootstrap_point_std": float(bs_std),
    }


def main():
    print("=" * 60)
    print("P1 PER-SEED SPEARMAN — ANALYSIS OF V1 LATENT ARTIFACTS")
    print("=" * 60)

    all_results = {}

    for mt, latents_dir in [("VAE", VAE_DIR), ("AE", AE_DIR)]:
        print(f"\n{'='*40}")
        print(f"  {mt} (dir: {latents_dir})")
        print(f"{'='*40}")

        seeds = detect_seeds(latents_dir)
        print(f"  Detected seeds: {seeds}")
        print(f"  V1 published seeds: {sorted(V1_PUBLISHED_SEEDS)}")

        all_d, all_a = [], []
        v1_d, v1_a = [], []
        per_seed_results = []

        for seed in seeds:
            seed_d, seed_a = [], []
            for cond, excluded in CONDITIONS.items():
                try:
                    pts = aspiration_for_seed(latents_dir, seed, cond, excluded)
                    for p in pts:
                        seed_d.append(p["d_before"])
                        seed_a.append(p["alpha"])
                except FileNotFoundError as e:
                    print(f"    SKIP {cond} seed {seed}: {e}")

            if len(seed_d) != EXPECTED_POINTS_PER_SEED:
                print(f"  seed {seed}: incomplete ({len(seed_d)}/{EXPECTED_POINTS_PER_SEED}), skipping")
                continue

            all_d.extend(seed_d)
            all_a.extend(seed_a)
            if seed in V1_PUBLISHED_SEEDS:
                v1_d.extend(seed_d)
                v1_a.extend(seed_a)

            d_arr = np.array(seed_d)
            a_arr = np.array(seed_a)
            rho, p = spearmanr(d_arr, a_arr)
            r_pearson, p_pearson = pearsonr(d_arr, a_arr)

            per_seed_results.append({
                "seed": seed,
                "in_v1_published": seed in V1_PUBLISHED_SEEDS,
                "n": len(d_arr),
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "mean_alpha": float(a_arr.mean()),
                "frac_aspirated": float((a_arr > 0).mean()),
            })

            v1_tag = " [V1]" if seed in V1_PUBLISHED_SEEDS else " [ext]"
            print(f"\n  seed {seed}{v1_tag} (n={len(d_arr)}):")
            print(f"    Spearman rho = {rho:.4f}, p = {p:.4f}")
            print(f"    Pearson r    = {r_pearson:.4f}, p = {p_pearson:.4f}")
            print(f"    mean alpha   = {a_arr.mean():.5f}")
            print(f"    aspirated    = {(a_arr > 0).sum()}/{len(a_arr)} ({100*(a_arr > 0).mean():.1f}%)")

        # Aggregates
        rhos_all = [r["spearman_rho"] for r in per_seed_results]
        rhos_v1 = [r["spearman_rho"] for r in per_seed_results if r["in_v1_published"]]

        print(f"\n  --- {mt} SUMMARY ---")
        print(f"  Per-seed rhos (all) : {[f'{r:.4f}' for r in rhos_all]}")
        print(f"  Per-seed rhos (V1)  : {[f'{r:.4f}' for r in rhos_v1]}")

        rho_all_mean = float(np.mean(rhos_all)) if rhos_all else 0.0
        rho_all_std = float(np.std(rhos_all)) if rhos_all else 0.0
        rho_v1_mean = float(np.mean(rhos_v1)) if rhos_v1 else 0.0
        rho_v1_std = float(np.std(rhos_v1)) if rhos_v1 else 0.0

        print(f"  Mean rho (all)      : {rho_all_mean:.4f} +/- {rho_all_std:.4f}")
        if rhos_v1:
            print(f"  Mean rho (V1)       : {rho_v1_mean:.4f} +/- {rho_v1_std:.4f}")

        global_all = compute_global(all_d, all_a)
        global_v1 = compute_global(v1_d, v1_a)

        print(f"  Global rho (all)    : {global_all['rho']:.4f}, p = {global_all['p']:.2e}, n = {global_all['n_points']}")
        print(f"  Global rho (V1)     : {global_v1['rho']:.4f}, p = {global_v1['p']:.2e}, n = {global_v1['n_points']}")
        print(f"  Bootstrap (all, pt) : {global_all['bootstrap_point_mean']:.4f} +/- {global_all['bootstrap_point_std']:.4f}")

        all_results[mt] = {
            "detected_seeds": seeds,
            "per_seed": per_seed_results,
            "aggregate_all_seeds": {
                "rho_mean": rho_all_mean,
                "rho_std": rho_all_std,
                "n_negative": sum(1 for r in rhos_all if r < 0),
                "n_seeds": len(rhos_all),
            },
            "aggregate_v1_seeds": {
                "rho_mean": rho_v1_mean,
                "rho_std": rho_v1_std,
                "n_negative": sum(1 for r in rhos_v1 if r < 0),
                "n_seeds": len(rhos_v1),
            },
            "global_all": global_all,
            "global_v1": global_v1,
        }

        if mt == "VAE":
            print(f"\n  --- P1 VERDICTS ({mt}) ---")
            verdict_v1, n_neg_v1 = compute_verdict(rhos_v1, "V1 published (3 seeds)", 2)
            verdict_all, n_neg_all = compute_verdict(rhos_all, "All seeds", 4)

            all_results[mt]["verdict_v1_published"] = verdict_v1
            all_results[mt]["verdict_all_seeds"] = verdict_all

            print(f"\n  METRIC:p1_verdict_v1={verdict_v1}")
            print(f"  METRIC:p1_verdict_all={verdict_all}")
            print(f"  METRIC:p1_rho_v1_mean={rho_v1_mean:.4f}")
            print(f"  METRIC:p1_rho_all_mean={rho_all_mean:.4f}")
            print(f"  METRIC:p1_global_v1_rho={global_v1['rho']:.4f}")
            print(f"  METRIC:p1_global_v1_p={global_v1['p']:.2e}")

    # Save JSON
    out = {
        "analysis": "P1 per-seed Spearman analysis",
        "artifact_status": "archived_v1_protocol_outputs",
        "note": "Latent files are archived outputs produced under the V1 experimental protocol. Seeds marked in_v1_published were included in the published ghost_aspiration.json; the others are additional archived seeds generated under the same V1 setup.",
        "latents_dir_vae": VAE_DIR,
        "latents_dir_ae": AE_DIR,
        "v1_published_seeds": sorted(V1_PUBLISHED_SEEDS),
        "results": all_results,
    }
    out_path = os.path.join(os.path.dirname(__file__), "p1_perseed_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
