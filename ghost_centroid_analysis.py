"""Ghost Centroid Aspiration Analysis.

Measures whether KL regularization causes surviving classes to be
"aspirated" toward the void left by removed classes.

Method:
1. Ghost centroid = centroid of removed class in FULL model
2. Procrustes-align ablated surviving centroids -> full surviving centroids
3. Measure displacement of each surviving class toward/away from ghost
4. Decompose: radial (toward ghost) vs tangential (lateral drift)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ExperimentConfig
from utils import compute_centroids, procrustes_align

CFG = ExperimentConfig()
BASE_DIR = CFG.base_dir
FIGURES_DIR = CFG.figures_dir
CONDITIONS = {k: v for k, v in CFG.conditions.items() if k != "full"}
SEEDS = CFG.seeds
ALL_CLASSES = list(range(10))
LATENT_DIRS = {"VAE": CFG.latents_dir, "AE": BASE_DIR / "ae_latents"}
COLORS = {"VAE": "#d62728", "AE": "#1f77b4"}
MARKERS = {"VAE": "o", "AE": "^"}


def aspiration_metrics(full_centroids, abl_centroids, excluded):
    """Procrustes-align ablated centroids, decompose displacement toward ghosts."""
    surviving = [c for c in ALL_CLASSES if c not in excluded]
    X_full = np.array([full_centroids[c] for c in surviving])
    X_abl = np.array([abl_centroids[c] for c in surviving])
    X_aligned, _, s, _ = procrustes_align(X_full, X_abl)
    residual = float(np.linalg.norm(X_aligned - X_full, axis=1).mean())

    results = []
    for gc in excluded:
        ghost = full_centroids[gc]
        for i, ci in enumerate(surviving):
            d_before = float(np.linalg.norm(X_full[i] - ghost))
            d_after = float(np.linalg.norm(X_aligned[i] - ghost))
            alpha = (d_before - d_after) / d_before if d_before > 1e-10 else 0.0
            disp = X_aligned[i] - X_full[i]
            dg = ghost - X_full[i]
            nd = np.linalg.norm(dg)
            if nd > 1e-10:
                dg /= nd
            rad = float(np.dot(disp, dg))
            results.append({
                "ghost": int(gc), "class": int(ci),
                "d_before": d_before, "d_after": d_after,
                "alpha": float(alpha), "radial": rad,
                "tangential": float(np.linalg.norm(disp - rad * dg)),
            })
    return results, float(s), residual


def load_centroids(latents_dir, cond, seed, classes):
    Z = np.load(latents_dir / f"{cond}_seed{seed}_latents.npy")
    L = np.load(latents_dir / f"{cond}_seed{seed}_labels.npy")
    return compute_centroids(Z, L, classes)


def scatter_with_trend(ax, x, y, mt):
    """Scatter plot with linear trend and Pearson r."""
    ax.scatter(x, y, c=COLORS[mt], marker=MARKERS[mt], alpha=0.45, s=45,
               label=mt, edgecolors="white", linewidth=0.5)
    z = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, np.polyval(z, xl), c=COLORS[mt], ls="--", lw=2, alpha=0.8,
            label=f"{mt} trend (r={pearsonr(x, y)[0]:.2f})")


def run():
    print("=" * 60 + "\nGHOST CENTROID ASPIRATION ANALYSIS\n" + "=" * 60)
    all_points, meta = {"VAE": [], "AE": []}, {"VAE": {}, "AE": {}}

    for mt in ["VAE", "AE"]:
        print(f"\n--- {mt} (latents: {LATENT_DIRS[mt]}) ---")
        for seed in SEEDS:
            full_c = load_centroids(LATENT_DIRS[mt], "full", seed, ALL_CLASSES)
            for cond, excluded in CONDITIONS.items():
                surviving = [c for c in ALL_CLASSES if c not in excluded]
                abl_c = load_centroids(LATENT_DIRS[mt], cond, seed, surviving)
                pts, ps, pr = aspiration_metrics(full_c, abl_c, excluded)
                for p in pts:
                    p.update(condition=cond, seed=seed)
                all_points[mt].extend(pts)
                meta[mt][f"{cond}_seed{seed}"] = {"procrustes_scale": ps, "procrustes_residual": pr}

    # ===== GLOBAL SUMMARY =====
    print(f"\n{'='*60}\nGLOBAL SUMMARY\n{'='*60}")
    summary = {}
    for mt in ["VAE", "AE"]:
        a = np.array([p["alpha"] for p in all_points[mt]])
        r = np.array([p["radial"] for p in all_points[mt]])
        d = np.array([p["d_before"] for p in all_points[mt]])
        r_da, p_da = pearsonr(d, a)
        rho_da, p_rho = spearmanr(d, a)
        r_dr, p_dr = pearsonr(d, r)
        print(f"\n  {mt} ({len(a)} pts): alpha={a.mean():+.4f}+/-{a.std():.4f}  "
              f"radial={r.mean():+.6f}+/-{r.std():.6f}")
        print(f"    Aspirated: {(a>0).sum()}/{len(a)} ({100*(a>0).mean():.1f}%)  "
              f"Radial>0: {(r>0).sum()}/{len(r)} ({100*(r>0).mean():.1f}%)")
        print(f"    Pearson(d,alpha): r={r_da:+.4f} p={p_da:.4f}  "
              f"Spearman: r={rho_da:+.4f} p={p_rho:.4f}  Pearson(d,rad): r={r_dr:+.4f}")
        summary[mt] = {
            "n_points": len(a), "mean_alpha": float(a.mean()), "std_alpha": float(a.std()),
            "mean_radial": float(r.mean()), "std_radial": float(r.std()),
            "frac_aspirated": float((a > 0).mean()),
            "pearson_d_alpha": {"r": float(r_da), "p": float(p_da)},
            "spearman_d_alpha": {"r": float(rho_da), "p": float(p_rho)},
            "pearson_d_radial": {"r": float(r_dr), "p": float(p_dr)},
        }

    # ===== PER-CONDITION =====
    print(f"\n{'='*60}\nPER-CONDITION (averaged across seeds)\n{'='*60}")
    per_condition = {"VAE": {}, "AE": {}}
    for mt in ["VAE", "AE"]:
        print(f"\n  --- {mt} ---")
        for cond, excluded in CONDITIONS.items():
            by_pair = defaultdict(list)
            for p in all_points[mt]:
                if p["condition"] == cond:
                    by_pair[(p["ghost"], p["class"])].append(p)
            avg_pts = [{"ghost": g, "class": c,
                        "d_before": np.mean([p["d_before"] for p in pl]),
                        "alpha": np.mean([p["alpha"] for p in pl]),
                        "radial": np.mean([p["radial"] for p in pl]),
                        "tangential": np.mean([p["tangential"] for p in pl])}
                       for (g, c), pl in by_pair.items()]
            alphas = [p["alpha"] for p in avg_pts]
            per_condition[mt][cond] = {
                "mean_alpha": float(np.mean(alphas)),
                "mean_radial": float(np.mean([p["radial"] for p in avg_pts])),
                "n_aspirated": sum(1 for a in alphas if a > 0), "n_total": len(alphas),
            }
            print(f"\n    {cond} (removed: {excluded}): alpha={np.mean(alphas):+.4f}  "
                  f"aspirated: {sum(1 for a in alphas if a > 0)}/{len(alphas)}")
            for g in excluded:
                g_pts = sorted([p for p in avg_pts if p["ghost"] == g], key=lambda x: x["d_before"])
                print(f"      Ghost {g}:")
                for p in g_pts:
                    print(f"        cls {p['class']:>1}: d={p['d_before']:.3f}  "
                          f"a={p['alpha']:+.4f}  r={p['radial']:+.5f}  t={p['tangential']:.5f}")

    # ===== PLOTS =====
    print(f"\n{'='*60}\nGENERATING PLOTS\n{'='*60}")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    pa = {mt: {k: np.array([p[k] for p in all_points[mt]])
               for k in ["d_before", "alpha", "radial", "tangential"]} for mt in ["VAE", "AE"]}

    for mt in ["VAE", "AE"]:
        scatter_with_trend(axes[0, 0], pa[mt]["d_before"], pa[mt]["alpha"], mt)
        scatter_with_trend(axes[0, 1], pa[mt]["d_before"], pa[mt]["radial"], mt)
        axes[1, 0].scatter(pa[mt]["radial"], pa[mt]["tangential"], c=COLORS[mt],
                           marker=MARKERS[mt], alpha=0.45, s=45, label=mt,
                           edgecolors="white", linewidth=0.5)
    for ax in [axes[0, 0], axes[0, 1]]:
        ax.axhline(0, color="gray", ls=":", alpha=0.5)
    axes[1, 0].axvline(0, color="gray", ls=":", alpha=0.5)
    axes[0, 0].set(xlabel="Distance to ghost", ylabel="Alpha", title="Classes toward void?")
    axes[0, 1].set(xlabel="Distance to ghost", ylabel="Radial disp.", title="Directed movement")
    axes[1, 0].set(xlabel="Radial", ylabel="Tangential", title="Aspiration vs drift")
    for ax in axes.flat[:3]:
        ax.legend(fontsize=9)

    ax = axes[1, 1]
    cond_names, x, w = list(CONDITIONS.keys()), np.arange(len(CONDITIONS)), 0.35
    for i, mt in enumerate(["VAE", "AE"]):
        ax.bar(x + i * w, [per_condition[mt][c]["mean_alpha"] for c in cond_names],
               w, label=mt, color=COLORS[mt], alpha=0.7)
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(cond_names, rotation=30, ha="right")
    ax.set(ylabel="Mean alpha", title="Aspiration by condition")
    ax.legend(fontsize=9)

    plt.suptitle("Ghost Centroid Aspiration: VAE vs AE\n(Procrustes alignment, no OOD encoding)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "ghost_aspiration.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fig_path}")
    plt.close()

    # ===== JSON =====
    out = {
        "method": "Procrustes-aligned ghost centroid aspiration",
        "description": ("Ghost = centroid of removed class in full model. "
                        "Alpha = (d_before - d_after) / d_before. "
                        "Radial = displacement projected toward ghost."),
        "summary": summary,
        "per_condition": {mt: dict(per_condition[mt]) for mt in ["VAE", "AE"]},
        "meta": meta,
    }
    out_path = BASE_DIR / "ghost_aspiration.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    run()
