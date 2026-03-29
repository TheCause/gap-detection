#!/usr/bin/env python3
"""Beta Sweep: KL dose-response curve for pressurized membrane prediction.

Trains VAEs at beta in {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}, measures W(H1),
ICD ratio, and aspiration gradient per condition. Seed 42, beta=1.0 reuses
existing latents when available.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json, time, numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ExperimentConfig
from vae import VAE
from utils import (select_device, load_dataset, train_model, extract_latents,
                   compute_centroids, procrustes_align)
from compare import bootstrap_compare, null_distribution

BETAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
CONDITIONS = {"full": [], "minus_7": [7], "minus_3_7": [3, 7], "minus_cluster": [3, 5, 8]}
SEED, ALL_CLASSES = 42, list(range(10))
BASE_DIR = Path(__file__).parent / "output"
SWEEP_DIR, EXISTING_LATENTS = BASE_DIR / "beta_sweep", BASE_DIR / "latents"


def _ghost_aspiration(full_c, abl_c, excluded):
    """How surviving centroids move toward ghost (removed class) positions."""
    surv = [c for c in ALL_CLASSES if c not in excluded]
    X_full = np.array([full_c[c] for c in surv])
    X_al, _, _, _ = procrustes_align(X_full, np.array([abl_c[c] for c in surv]))
    pts = []
    for gc in excluded:
        g = full_c[gc]
        for i in range(len(surv)):
            db = np.linalg.norm(X_full[i] - g)
            da = np.linalg.norm(X_al[i] - g)
            alpha = (db - da) / db if db > 1e-10 else 0.0
            disp = X_al[i] - X_full[i]; dg = g - X_full[i]; nd = np.linalg.norm(dg)
            pts.append({"d_before": float(db), "alpha": float(alpha),
                        "radial": float(np.dot(disp, dg/nd)) if nd > 1e-10 else 0.0})
    return pts


def _icd_ratio(full_c, abl_c, surviving):
    """Inter-centroid distance ratio (ablated / full)."""
    fd = [np.linalg.norm(full_c[ci]-full_c[cj]) for i,ci in enumerate(surviving) for j,cj in enumerate(surviving) if i<j]
    ad = [np.linalg.norm(abl_c[ci]-abl_c[cj]) for i,ci in enumerate(surviving) for j,cj in enumerate(surviving) if i<j]
    return np.mean(ad)/np.mean(fd) if np.mean(fd) > 0 else 0.0


def _load_or_train(config, device, beta, cond, excluded, beta_dir):
    """Load cached latents, reuse beta=1.0 existing, or train fresh."""
    cache_z, cache_l = beta_dir / f"{cond}_latents.npy", beta_dir / f"{cond}_labels.npy"
    if cache_z.exists():
        return np.load(cache_z), np.load(cache_l)
    if beta == 1.0:
        src = EXISTING_LATENTS / f"{cond}_seed{SEED}_latents.npy"
        if src.exists():
            Z, L = np.load(src), np.load(src.with_name(f"{cond}_seed{SEED}_labels.npy"))
            np.save(cache_z, Z); np.save(cache_l, L)
            return Z, L
    torch.manual_seed(SEED); np.random.seed(SEED)
    train_ds = load_dataset("mnist", train=True, excluded_classes=excluded or None)
    test_ds = load_dataset("mnist", train=False, excluded_classes=excluded or None)
    model = VAE(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(device)
    train_model(model, train_ds, config, device, label=f"{cond}/b{beta}", beta=beta)
    Z, L = extract_latents(model, DataLoader(test_ds, batch_size=512, shuffle=False), device)
    np.save(cache_z, Z); np.save(cache_l, L)
    torch.save(model.state_dict(), beta_dir / f"{cond}.pt")
    return Z, L


def run():
    config = ExperimentConfig()
    device = select_device()
    print(f"BETA SWEEP | betas={BETAS} | B={config.n_bootstrap} | n={config.tda_n_samples} | {device}")
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}

    for beta in BETAS:
        tb = time.time()
        bk = f"beta_{beta}"
        bd = SWEEP_DIR / bk; bd.mkdir(exist_ok=True)
        print(f"\n--- BETA={beta} ---")

        latents = {c: _load_or_train(config, device, beta, c, ex, bd)
                   for c, ex in CONDITIONS.items()}
        Z_full = latents["full"][0]

        # TDA bootstrap
        w_h1 = {}
        for cond in [c for c in CONDITIONS if c != "full"]:
            comp = bootstrap_compare(Z_full, latents[cond][0],
                                     n_samples=config.tda_n_samples, n_bootstrap=config.n_bootstrap)
            w = comp["wasserstein_H1"]
            w_h1[cond] = {k: w[k] for k in ("mean", "std", "ci_low", "ci_high")}
            print(f"  W(H1) {cond}: {w['mean']:.2f} [{w['ci_low']:.2f}, {w['ci_high']:.2f}]")
        null = null_distribution(Z_full, n_samples=config.tda_n_samples, n_bootstrap=config.n_bootstrap)
        null_95 = null["null_wasserstein_H1"]["ci_high_95"]

        # Ghost centroid + ICD
        full_c = compute_centroids(*latents["full"], ALL_CLASSES)
        all_asp, icds = [], {}
        for cond, excluded in [(c, e) for c, e in CONDITIONS.items() if c != "full"]:
            surv = [c for c in ALL_CLASSES if c not in excluded]
            abl_c = compute_centroids(*latents[cond], surv)
            icds[cond] = float(_icd_ratio(full_c, abl_c, surv))
            all_asp.extend(_ghost_aspiration(full_c, abl_c, excluded))


        d_b = np.array([p["d_before"] for p in all_asp])
        alphas = np.array([p["alpha"] for p in all_asp])
        r_asp, p_asp = pearsonr(d_b, alphas) if len(d_b) > 2 else (0.0, 1.0)
        print(f"  Aspiration: r={r_asp:.4f} p={p_asp:.4f} mean_alpha={alphas.mean():+.4f}")

        results[bk] = {"beta": beta, "w_h1": w_h1, "null_95": float(null_95), "icd_ratios": icds,
                        "aspiration": {"pearson_r": float(r_asp), "pearson_p": float(p_asp),
                                       "mean_alpha": float(alphas.mean()), "n_points": len(alphas)},
                        "time_s": time.time() - tb}

    print(f"\nTOTAL: {time.time()-t0:.0f}s")
    with open(SWEEP_DIR / "beta_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    _plot(results, config)
    _summary(results)


def _plot(results, config):
    betas_s = sorted(BETAS)
    conds = ["minus_7", "minus_3_7", "minus_cluster"]
    cc = {"minus_7": "#e41a1c", "minus_3_7": "#377eb8", "minus_cluster": "#4daf4a"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for c in conds:  # W(H1)
        v = [results[f"beta_{b}"]["w_h1"][c] for b in betas_s]
        axes[0,0].plot(betas_s, [x["mean"] for x in v], "o-", color=cc[c], label=c, lw=2)
        axes[0,0].fill_between(betas_s, [x["ci_low"] for x in v], [x["ci_high"] for x in v], alpha=.12, color=cc[c])
    axes[0,0].plot(betas_s, [results[f"beta_{b}"]["null_95"] for b in betas_s], "k--", alpha=.5, label="Null 95th")
    axes[0,0].set(xscale="log", xlabel="beta", ylabel="W(H1)", title="Topological signal vs KL pressure")
    axes[0,0].legend(fontsize=9)

    for c in conds:  # ICD ratio
        axes[0,1].plot(betas_s, [results[f"beta_{b}"]["icd_ratios"][c] for b in betas_s], "o-", color=cc[c], label=c, lw=2)
    axes[0,1].axhline(1.0, color="gray", ls=":")
    axes[0,1].set(xscale="log", xlabel="beta", ylabel="ICD ratio", title="Contraction vs KL pressure")
    axes[0,1].legend(fontsize=9)

    r_v = [results[f"beta_{b}"]["aspiration"]["pearson_r"] for b in betas_s]
    p_v = [results[f"beta_{b}"]["aspiration"]["pearson_p"] for b in betas_s]
    axes[1,0].bar(range(len(betas_s)), r_v, color=["#d62728" if p<.05 else "#999" for p in p_v], alpha=.7)
    axes[1,0].set_xticks(range(len(betas_s))); axes[1,0].set_xticklabels([str(b) for b in betas_s])
    axes[1,0].axhline(0, color="gray", ls=":")
    axes[1,0].set(xlabel="beta", ylabel="Pearson r", title="Aspiration gradient (red=p<0.05)")

    axes[1,1].plot(betas_s, [results[f"beta_{b}"]["aspiration"]["mean_alpha"] for b in betas_s], "o-", color="#d62728", lw=2)
    axes[1,1].axhline(0, color="gray", ls=":")
    axes[1,1].set(xscale="log", xlabel="beta", ylabel="Mean alpha", title="Mean aspiration vs KL pressure")

    plt.suptitle(f"Beta Sweep (seed={SEED}, B={ExperimentConfig().n_bootstrap})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(config.figures_dir / "beta_sweep.png", dpi=150, bbox_inches="tight"); plt.close()


def _summary(results):
    betas_s = sorted(BETAS)
    print(f"\n{'='*60}\nDOSE-RESPONSE SUMMARY\n{'='*60}")
    hdr = f"{'beta':>6} | {'W m7':>8} | {'W m37':>8} | {'W mc':>8} | {'Null95':>7} | {'ICD m7':>7} | {'ICD mc':>7} | {'Asp r':>7} | {'Asp p':>8}"
    print(hdr); print("-" * len(hdr))
    for b in betas_s:
        r = results[f"beta_{b}"]
        print(f"{b:>6.1f} | {r['w_h1']['minus_7']['mean']:>8.2f} | {r['w_h1']['minus_3_7']['mean']:>8.2f} | "
              f"{r['w_h1']['minus_cluster']['mean']:>8.2f} | {r['null_95']:>7.2f} | "
              f"{r['icd_ratios']['minus_7']:>7.4f} | {r['icd_ratios']['minus_cluster']:>7.4f} | "
              f"{r['aspiration']['pearson_r']:>7.4f} | {r['aspiration']['pearson_p']:>8.4f}")


if __name__ == "__main__":
    run()
