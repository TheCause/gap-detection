#!/usr/bin/env python3
"""OOD Reconstruction Error Baseline.

For each gap condition, train a VAE on the training set (with excluded classes),
then measure reconstruction MSE on:
  - In-distribution test samples (classes seen during training)
  - OOD test samples (excluded classes the VAE never saw)

Compare gap detection power: MSE-based vs TDA (Wasserstein H1).
Bootstrap CIs for all metrics.
"""

import json
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vae import VAE
from config import ExperimentConfig


from utils import select_device


def get_class_subsets(dataset_name="mnist", data_root="./data"):
    """Load full test set, return (dataset, targets tensor)."""
    ds_cls = datasets.FashionMNIST if dataset_name == "fashion_mnist" else datasets.MNIST
    ds = ds_cls(root=data_root, train=False, download=True, transform=transforms.ToTensor())
    targets = torch.tensor(ds.targets) if not isinstance(ds.targets, torch.Tensor) else ds.targets
    return ds, targets


def compute_mse_per_sample(model, dataloader, device):
    """Compute per-sample reconstruction MSE."""
    model.eval()
    mses = []
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            recon, _, _ = model(batch_x)
            mse = F.mse_loss(recon, batch_x, reduction='none')
            mse = mse.view(mse.size(0), -1).mean(dim=1)  # per-sample MSE
            mses.append(mse.cpu().numpy())
    return np.concatenate(mses)


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    means = np.sort(means)
    lo = (1 - ci) / 2
    hi = 1 - lo
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_low": float(means[int(lo * n_bootstrap)]),
        "ci_high": float(means[int(hi * n_bootstrap)]),
        "n": len(values),
    }


def auroc_mse(in_dist_mse, ood_mse):
    """Compute AUROC for MSE-based OOD detection (higher MSE = OOD)."""
    labels = np.concatenate([np.zeros(len(in_dist_mse)), np.ones(len(ood_mse))])
    scores = np.concatenate([in_dist_mse, ood_mse])
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sorted_idx = np.argsort(-scores)  # descending
    sorted_labels = labels[sorted_idx]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    tpr = tp / n_pos
    fpr = fp / n_neg
    auroc = np.trapezoid(tpr, fpr)
    return float(auroc)


def bootstrap_auroc(in_dist_mse, ood_mse, n_bootstrap=200, seed=42):
    """Bootstrap AUROC with CI."""
    rng = np.random.RandomState(seed)
    aurocs = []
    for _ in range(n_bootstrap):
        in_sample = rng.choice(in_dist_mse, size=len(in_dist_mse), replace=True)
        ood_sample = rng.choice(ood_mse, size=len(ood_mse), replace=True)
        aurocs.append(auroc_mse(in_sample, ood_sample))
    aurocs = np.sort(aurocs)
    return {
        "mean": float(np.mean(aurocs)),
        "ci_low": float(aurocs[int(0.025 * n_bootstrap)]),
        "ci_high": float(aurocs[int(0.975 * n_bootstrap)]),
    }


def train_vae_for_condition(config, condition, seed, device):
    """Train VAE, return model. Reuse cached model if available."""
    run = config.run_name(condition, seed)
    model_path = config.models_dir / f"{run}.pt"

    model = VAE(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(device)

    if model_path.exists():
        print(f"  [{run}] Loading cached model")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        return model

    # Train
    from vae import vae_loss
    torch.manual_seed(seed)
    np.random.seed(seed)
    excluded = config.conditions[condition]
    ds_cls = datasets.FashionMNIST if config.dataset_name == "fashion_mnist" else datasets.MNIST
    dataset = ds_cls(root="./data", train=True, download=True, transform=transforms.ToTensor())
    targets = torch.tensor(dataset.targets) if not isinstance(dataset.targets, torch.Tensor) else dataset.targets

    if excluded:
        mask = torch.tensor([t not in excluded for t in targets])
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss, _, _ = vae_loss(recon, batch_x, mu, logvar, beta=config.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(dataset)
            print(f"  [{run}] Epoch {epoch+1}/{config.epochs} - Loss: {avg:.2f}")

    config.ensure_dirs()
    torch.save(model.state_dict(), model_path)
    print(f"  [{run}] Trained and saved")
    return model


def run(dataset_name="mnist"):
    config = ExperimentConfig()
    config.dataset_name = dataset_name
    if dataset_name == "fashion_mnist":
        config.base_dir = Path(__file__).parent / "output_fashion_mnist"
    config.ensure_dirs()
    device = select_device()

    print(f"=== OOD Reconstruction Error Baseline ===")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print()

    t0 = time.time()

    # Load full test set
    test_ds, test_targets = get_class_subsets(dataset_name)

    results = {}

    for condition, excluded in config.conditions.items():
        if condition == "full" or not excluded:
            continue

        print(f"\n--- Condition: {condition} (excluded: {excluded}) ---")
        condition_results = {"excluded": excluded, "seeds": {}}

        for seed in config.seeds:
            model = train_vae_for_condition(config, condition, seed, device)

            # In-distribution: test samples from seen classes
            in_mask = torch.tensor([int(t) not in excluded for t in test_targets])
            in_indices = in_mask.nonzero(as_tuple=True)[0].tolist()
            in_loader = DataLoader(Subset(test_ds, in_indices), batch_size=256, shuffle=False)

            # OOD: test samples from excluded classes
            ood_mask = torch.tensor([int(t) in excluded for t in test_targets])
            ood_indices = ood_mask.nonzero(as_tuple=True)[0].tolist()
            ood_loader = DataLoader(Subset(test_ds, ood_indices), batch_size=256, shuffle=False)

            in_mse = compute_mse_per_sample(model, in_loader, device)
            ood_mse = compute_mse_per_sample(model, ood_loader, device)

            aur = auroc_mse(in_mse, ood_mse)
            print(f"  seed{seed}: in_MSE={np.mean(in_mse):.6f} | ood_MSE={np.mean(ood_mse):.6f} | "
                  f"ratio={np.mean(ood_mse)/np.mean(in_mse):.3f} | AUROC={aur:.4f}")

            condition_results["seeds"][seed] = {
                "in_mse": bootstrap_ci(in_mse),
                "ood_mse": bootstrap_ci(ood_mse),
                "auroc": aur,
            }

        # Aggregate across seeds
        all_in = [condition_results["seeds"][s]["in_mse"]["mean"] for s in config.seeds]
        all_ood = [condition_results["seeds"][s]["ood_mse"]["mean"] for s in config.seeds]
        all_auroc = [condition_results["seeds"][s]["auroc"] for s in config.seeds]

        # Bootstrap AUROC from first seed (full sample)
        model = train_vae_for_condition(config, condition, config.seeds[0], device)
        in_mse_full = compute_mse_per_sample(model, in_loader, device)
        ood_mse_full = compute_mse_per_sample(model, ood_loader, device)
        auroc_boot = bootstrap_auroc(in_mse_full, ood_mse_full)

        condition_results["aggregate"] = {
            "in_mse_mean": float(np.mean(all_in)),
            "in_mse_std": float(np.std(all_in)),
            "ood_mse_mean": float(np.mean(all_ood)),
            "ood_mse_std": float(np.std(all_ood)),
            "ratio_mean": float(np.mean(all_ood) / np.mean(all_in)),
            "auroc_mean": float(np.mean(all_auroc)),
            "auroc_std": float(np.std(all_auroc)),
            "auroc_bootstrap": auroc_boot,
        }

        results[condition] = condition_results

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Condition':<15} {'In-MSE':<12} {'OOD-MSE':<12} {'Ratio':<8} {'AUROC':<12} {'AUROC CI'}")
    print(f"{'='*70}")
    for cond, data in results.items():
        agg = data["aggregate"]
        aboot = agg["auroc_bootstrap"]
        print(f"{cond:<15} {agg['in_mse_mean']:.6f}    {agg['ood_mse_mean']:.6f}    "
              f"{agg['ratio_mean']:.3f}   {agg['auroc_mean']:.4f}      "
              f"[{aboot['ci_low']:.4f}, {aboot['ci_high']:.4f}]")

    total = time.time() - t0
    print(f"\nDone in {total:.0f}s")

    # Save
    output = {
        "dataset": dataset_name,
        "results": results,
        "timing_s": total,
        "interpretation": (
            "AUROC measures MSE-based OOD detection power. "
            "AUROC ~0.5 = MSE cannot detect the gap. "
            "AUROC ~1.0 = MSE fully detects the gap. "
            "Compare with TDA W_H1 significance to assess complementarity."
        ),
    }
    out_path = config.base_dir / "ood_recon_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    run(ds)
