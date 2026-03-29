"""Visualization: persistence diagrams, UMAP, comparisons."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # M4 headless
import matplotlib.pyplot as plt
from pathlib import Path


def plot_persistence_diagrams(all_diagrams: dict, figures_dir: Path):
    """Side-by-side persistence diagrams for all conditions."""
    conditions = list(all_diagrams.keys())
    n = len(conditions)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for i, cond in enumerate(conditions):
        diagrams = all_diagrams[cond]
        for dim in range(2):
            ax = axes[dim, i] if n > 1 else axes[dim]
            dgm = diagrams[dim]
            finite = dgm[dgm[:, 1] < np.inf]
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                ax.scatter(finite[:, 0], finite[:, 1], s=10, alpha=0.6,
                           c=lifetimes, cmap='viridis')
            # Diagonal
            lim = ax.get_xlim()[1] if len(finite) > 0 else 1
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
            ax.set_title(f"{cond} — H{dim}")
            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")

    plt.tight_layout()
    plt.savefig(figures_dir / "persistence_diagrams.png", dpi=150)
    plt.close()
    print("  Saved persistence_diagrams.png")


def plot_umap_latents(all_latents: dict, all_labels: dict, figures_dir: Path):
    """UMAP projection of latent spaces, colored by class."""
    try:
        import umap
    except ImportError:
        print("  UMAP not available, skipping latent space plot")
        return

    conditions = list(all_latents.keys())
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, cond in enumerate(conditions):
        latents = all_latents[cond]
        labels = all_labels[cond]

        # Subsample for speed
        if len(latents) > 5000:
            idx = np.random.choice(len(latents), 5000, replace=False)
            latents = latents[idx]
            labels = labels[idx]

        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(latents)

        scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1],
                                  c=labels, cmap='tab10', s=2, alpha=0.5,
                                  vmin=0, vmax=9)
        axes[i].set_title(f"{cond} (n={len(latents)})")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.colorbar(scatter, ax=axes, label="Digit class", shrink=0.8)
    plt.tight_layout()
    plt.savefig(figures_dir / "umap_latents.png", dpi=150)
    plt.close()
    print("  Saved umap_latents.png")


def plot_betti_comparison(results: dict, figures_dir: Path):
    """Barplot of Betti numbers per condition."""
    conditions = []
    h0_means, h0_stds = [], []
    h1_means, h1_stds = [], []

    for cond, data in results.items():
        if "tda_features" not in data:
            continue
        conditions.append(cond)
        feats = data["tda_features"]
        h0_means.append(feats.get("H0_count_mean", 0))
        h0_stds.append(feats.get("H0_count_std", 0))
        h1_means.append(feats.get("H1_count_mean", 0))
        h1_stds.append(feats.get("H1_count_std", 0))

    x = np.arange(len(conditions))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, h0_means, width, yerr=h0_stds, capsize=5, label='H0 (components)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=30)
    ax1.set_ylabel("Betti H0")
    ax1.set_title("Connected Components")

    ax2.bar(x, h1_means, width, yerr=h1_stds, capsize=5, color='orange', label='H1 (loops)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=30)
    ax2.set_ylabel("Betti H1")
    ax2.set_title("1-Dimensional Holes")

    plt.tight_layout()
    plt.savefig(figures_dir / "betti_comparison.png", dpi=150)
    plt.close()
    print("  Saved betti_comparison.png")


def plot_wasserstein_matrix(comparisons: dict, figures_dir: Path):
    """Heatmap of Wasserstein distances between conditions."""
    conditions = sorted(set(
        c.split("_vs_")[0] for c in comparisons.keys() if "_vs_" in c
    ) | set(
        c.split("_vs_")[1] for c in comparisons.keys() if "_vs_" in c
    ))
    n = len(conditions)
    if n == 0:
        print("  No comparisons to plot")
        return

    matrix_h0 = np.zeros((n, n))
    matrix_h1 = np.zeros((n, n))

    cond_idx = {c: i for i, c in enumerate(conditions)}
    for key, data in comparisons.items():
        if "_vs_" not in key:
            continue
        a, b = key.split("_vs_")
        if a in cond_idx and b in cond_idx:
            i, j = cond_idx[a], cond_idx[b]
            matrix_h0[i, j] = data.get("wasserstein_H0", {}).get("mean", 0)
            matrix_h0[j, i] = matrix_h0[i, j]
            matrix_h1[i, j] = data.get("wasserstein_H1", {}).get("mean", 0)
            matrix_h1[j, i] = matrix_h1[i, j]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(matrix_h0, cmap='YlOrRd')
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(conditions, rotation=45, ha='right')
    ax1.set_yticklabels(conditions)
    ax1.set_title("Wasserstein H0")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(matrix_h1, cmap='YlOrRd')
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(conditions, rotation=45, ha='right')
    ax2.set_yticklabels(conditions)
    ax2.set_title("Wasserstein H1")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Annotate cells
    for ax, mat in [(ax1, matrix_h0), (ax2, matrix_h1)]:
        for ii in range(n):
            for jj in range(n):
                if mat[ii, jj] > 0:
                    ax.text(jj, ii, f"{mat[ii,jj]:.2f}", ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / "wasserstein_matrix.png", dpi=150)
    plt.close()
    print("  Saved wasserstein_matrix.png")


def generate_report(results: dict, comparisons: dict, null_stats: dict,
                    output_dir: Path):
    """Generate markdown report."""
    lines = [
        "# Experience Fondatrice : Detection de Lacunes via Homologie Persistante",
        "",
        "## Design experimental",
        "- **Modele** : VAE convolutionnel (latent_dim=16)",
        "- **Dataset** : MNIST",
        "- **Conditions** : full (10 classes), minus_7, minus_3_7, minus_cluster",
        "- **Seeds** : 3 par condition (reproductibilite)",
        "",
        "## Resultats par condition",
        "",
    ]

    for cond, data in results.items():
        feats = data.get("tda_features", {})
        lines.append(f"### {cond}")
        lines.append(f"- Classes exclues : {data.get('excluded', [])}")
        lines.append(f"- H0 count : {feats.get('H0_count_mean', 'N/A'):.1f} +/- {feats.get('H0_count_std', 0):.1f}")
        lines.append(f"- H1 count : {feats.get('H1_count_mean', 'N/A'):.1f} +/- {feats.get('H1_count_std', 0):.1f}")
        lines.append(f"- H1 total persistence : {feats.get('H1_total_persistence_mean', 'N/A'):.3f} +/- {feats.get('H1_total_persistence_std', 0):.3f}")
        lines.append("")

    lines.append("## Comparaisons (Wasserstein)")
    lines.append("")
    lines.append("| Comparaison | W_H0 (mean +/- std) | W_H1 (mean +/- std) | p(W_H1 > 0) |")
    lines.append("|---|---|---|---|")

    for key, data in comparisons.items():
        if "_vs_" not in key:
            continue
        wh0 = data.get("wasserstein_H0", {})
        wh1 = data.get("wasserstein_H1", {})
        lines.append(
            f"| {key} | {wh0.get('mean', 0):.3f} +/- {wh0.get('std', 0):.3f} "
            f"| {wh1.get('mean', 0):.3f} +/- {wh1.get('std', 0):.3f} "
            f"| {wh1.get('p_gt_zero', 0):.2f} |"
        )

    lines.append("")

    if null_stats:
        nh1 = null_stats.get("null_wasserstein_H1", {})
        lines.append("## Distribution nulle (resample interne)")
        lines.append(f"- W_H1 null : mean={nh1.get('mean', 0):.3f}, 95th={nh1.get('ci_high_95', 0):.3f}, 99th={nh1.get('ci_high_99', 0):.3f}")
        lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")

    # Auto-detect success
    sig_comparisons = []
    for key, data in comparisons.items():
        if "_vs_" not in key or "full" not in key:
            continue
        wh1 = data.get("wasserstein_H1", {})
        if wh1.get("ci_low", 0) > 0:
            sig_comparisons.append(key)

    if sig_comparisons:
        lines.append(f"**SUCCES** : {len(sig_comparisons)} comparaison(s) montrent une difference significative "
                      f"en Wasserstein H1 (CI 95% > 0).")
        for c in sig_comparisons:
            lines.append(f"- {c}")
        lines.append("")
        lines.append("La conjecture est soutenue : les lacunes du corpus creent des trous topologiques "
                      "detectables dans l'espace latent.")
    else:
        lines.append("**ECHEC/PARTIEL** : Aucune comparaison ne montre de difference significative en H1.")
        lines.append("Pistes : augmenter n_samples, changer architecture, tester avec autoencoder deterministe.")

    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report)
    print(f"  Report saved to {output_dir / 'report.md'}")
    return report
