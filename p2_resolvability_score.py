#!/usr/bin/env python3
"""P2 — Geometric resolvability score per condition x beta.

Empirical class-conditional GMM (no EM fitting — uses labels directly).
Metric defined BEFORE looking at results (see p2_metric_definition.md).

Compares beta*_GMM(cond) to beta*_TDA(cond).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import numpy as np
from itertools import combinations

# === PATHS ===
BETA_DIR = "/Users/regis/dev/epistemologue/output/experiment/beta_sweep"
BETAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

CONDITIONS = {
    "minus_cluster": {"excluded": [3, 5, 8], "beta_star_tda": 2.0},
    "minus_3_7": {"excluded": [3, 7], "beta_star_tda": 2.0},
    "minus_7": {"excluded": [7], "beta_star_tda": 5.0},
}

ALL_CLASSES = list(range(10))


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return path


def empirical_class_gmm(Z, labels, classes):
    """Compute empirical class-conditional GMM parameters from labels.
    No EM — centroids, covariances, and weights directly from labeled data.
    """
    centroids = {}
    sigmas = {}
    weights = {}
    counts = {}
    d = Z.shape[1]
    N = len(labels)

    for c in classes:
        mask = labels == c
        n_c = mask.sum()
        if n_c < 2:
            continue
        Zc = Z[mask]
        centroids[c] = Zc.mean(axis=0)
        cov = np.atleast_2d(np.cov(Zc, rowvar=False))
        sigmas[c] = np.sqrt(np.trace(cov) / d)
        weights[c] = n_c / N
        counts[c] = int(n_c)

    return centroids, sigmas, weights, counts


def resolvability_score(centroids, sigmas, weights):
    """Compute geometric resolvability = min_separation / weighted_spread."""
    classes = sorted(centroids.keys())
    if len(classes) < 2:
        return 0.0, 0.0, 0.0, None

    # Separation = min pairwise distance, track the pair
    min_dist = float("inf")
    min_pair = None
    for ci, cj in combinations(classes, 2):
        d = np.linalg.norm(centroids[ci] - centroids[cj])
        if d < min_dist:
            min_dist = d
            min_pair = (int(ci), int(cj))
    separation = min_dist

    # Spread = weighted mean effective std
    spread = sum(weights[c] * sigmas[c] for c in classes)

    score = separation / spread if spread > 1e-10 else 0.0

    return separation, spread, score, min_pair


def run_condition(name, latents_key, classes, beta_star_tda):
    """Run P2 analysis for one condition across all betas."""
    K = len(classes)
    print(f"\n{'='*50}")
    print(f"  {name} (K={K})")
    print(f"  beta*_TDA = {beta_star_tda}")
    print(f"{'='*50}")

    cond_results = []

    for beta in BETAS:
        beta_dir = os.path.join(BETA_DIR, f"beta_{beta}")
        latents_path = os.path.join(beta_dir, f"{latents_key}_latents.npy")
        labels_path = os.path.join(beta_dir, f"{latents_key}_labels.npy")

        try:
            Z = np.load(require(latents_path))
            L = np.load(require(labels_path))
        except FileNotFoundError as e:
            print(f"  beta={beta}: SKIP ({e})")
            continue

        # Verify labels
        present = sorted(set(int(x) for x in L))
        if present != classes:
            print(f"  beta={beta}: LABEL MISMATCH present={present} expected={classes}")
            continue

        centroids, sigmas, weights, counts = empirical_class_gmm(Z, L, classes)
        separation, spread, score, min_pair = resolvability_score(centroids, sigmas, weights)

        cond_results.append({
            "beta": beta,
            "separation": float(separation),
            "spread": float(spread),
            "score": float(score),
            "min_pair": min_pair,
            "n_points": len(Z),
            "K": len(centroids),
            "counts_per_class": counts,
        })

        pair_str = f"{min_pair[0]}-{min_pair[1]}" if min_pair else "?"
        print(f"  beta={beta:5.1f}: sep={separation:.4f}  spread={spread:.4f}  score={score:.4f}  pair={pair_str}  n={len(Z)}")

    return cond_results


def main():
    print("=" * 60)
    print("P2 — GEOMETRIC RESOLVABILITY SCORE")
    print("Empirical class-conditional GMM (label-based, no EM)")
    print("=" * 60)

    results = {}

    # Ablated conditions (verdict principal)
    for cond, info in CONDITIONS.items():
        excluded = info["excluded"]
        surviving = sorted(set(ALL_CLASSES) - set(excluded))
        beta_star_tda = info["beta_star_tda"]

        cond_results = run_condition(cond, cond, surviving, beta_star_tda)

        if not cond_results:
            print(f"  NO RESULTS for {cond}")
            continue

        best = max(cond_results, key=lambda x: x["score"])
        beta_star_gmm = best["beta"]

        ratio = max(beta_star_gmm, beta_star_tda) / min(beta_star_gmm, beta_star_tda)
        if ratio <= 2.0:
            verdict = "PASS"
        elif ratio <= 3.0:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"

        print(f"\n  beta*_GMM  = {beta_star_gmm} (score={best['score']:.4f})")
        print(f"  beta*_TDA  = {beta_star_tda}")
        print(f"  ratio      = {ratio:.2f}")
        print(f"  VERDICT    = {verdict}")

        results[cond] = {
            "per_beta": cond_results,
            "beta_star_gmm": beta_star_gmm,
            "beta_star_tda": beta_star_tda,
            "ratio": float(ratio),
            "verdict": verdict,
        }

    # Full dataset (diagnostic secondaire — pas de verdict)
    full_classes = ALL_CLASSES
    full_results = run_condition("full (diagnostic)", "full", full_classes, None)
    if full_results:
        best_full = max(full_results, key=lambda x: x["score"])
        print(f"\n  full beta*_GMM = {best_full['beta']} (score={best_full['score']:.4f})")
        results["full_diagnostic"] = {
            "per_beta": full_results,
            "beta_star_gmm": best_full["beta"],
            "note": "diagnostic only, no verdict",
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Condition':<18} {'beta*_GMM':>10} {'beta*_TDA':>10} {'Ratio':>8} {'Verdict':>10}")
    for cond in CONDITIONS:
        if cond in results:
            r = results[cond]
            print(f"  {cond:<18} {r['beta_star_gmm']:>10.1f} {r['beta_star_tda']:>10.1f} {r['ratio']:>8.2f} {r['verdict']:>10}")
    if "full_diagnostic" in results:
        r = results["full_diagnostic"]
        print(f"  {'full (diag)':<18} {r['beta_star_gmm']:>10.1f} {'—':>10} {'—':>8} {'—':>10}")

    # Metrics
    n_pass = sum(1 for c in CONDITIONS if c in results and results[c]["verdict"] == "PASS")
    print(f"\n  METRIC:p2_n_pass={n_pass}")
    print(f"  METRIC:p2_n_total={len(CONDITIONS)}")
    for cond in CONDITIONS:
        if cond in results:
            r = results[cond]
            print(f"  METRIC:p2_{cond}_gmm={r['beta_star_gmm']}")
            print(f"  METRIC:p2_{cond}_tda={r['beta_star_tda']}")
            print(f"  METRIC:p2_{cond}_verdict={r['verdict']}")

    # Save
    out = {
        "analysis": "P2 geometric resolvability score",
        "method": "empirical class-conditional GMM (label-based, no EM)",
        "metric": "separation / weighted_spread where separation=min inter-centroid dist, spread=sum(pi_k * sigma_k)",
        "artifact_status": "archived_v1_protocol_outputs",
        "betas": BETAS,
        "results": results,
    }
    out_path = os.path.join(os.path.dirname(__file__), "p2_resolvability_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
