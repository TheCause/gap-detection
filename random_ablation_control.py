#!/usr/bin/env python3
"""Random ablation control (W5) — paired design.

For each seed, trains THREE models with the same reference:
  1. VAE full (100% data, all classes) — reference
  2. VAE random (same fraction as ablation, all classes kept)
  3. VAE categorical (same fraction, specific classes removed) — loaded from cache

Computes W(H1) for random->full and categorical->full using the SAME full model.
Reports Delta = categorical - random per seed to isolate topological signal
from dataset size effect.
"""

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vae import VAE, vae_loss
from tda import compute_persistence, compute_wasserstein
from config import ExperimentConfig


from utils import select_device


def train_vae_fresh(config, dataset, seed, device, label=""):
    """Train VAE on given dataset subset."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    model = VAE(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(device)
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
            print(f"  [{label}] Epoch {epoch+1}/{config.epochs} - Loss: {avg:.2f}")

    return model


def extract_latents(model, test_loader, device):
    """Extract mu vectors from test set."""
    model.eval()
    all_mu = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            mu, _ = model.encode(batch_x)
            all_mu.append(mu.cpu().numpy())
    return np.concatenate(all_mu)


def run(dataset_name="mnist"):
    config = ExperimentConfig()
    config.dataset_name = dataset_name
    if dataset_name == "fashion_mnist":
        config.base_dir = Path(__file__).parent / "output_fashion_mnist"
    config.ensure_dirs()
    device = select_device()

    print(f"=== Random Ablation Control (paired design) ===")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print()

    t0 = time.time()

    # Load datasets
    ds_cls = datasets.FashionMNIST if dataset_name == "fashion_mnist" else datasets.MNIST
    train_ds = ds_cls(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_ds = ds_cls(root="./data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    train_targets = torch.tensor(train_ds.targets) if not isinstance(train_ds.targets, torch.Tensor) else train_ds.targets

    # Conditions to test (categorical ablation name -> excluded classes, fraction)
    conditions = {
        "minus_7": {"excluded": [7], "fraction": 0.90},
        "minus_3_7": {"excluded": [3, 7], "fraction": 0.80},
        "minus_2_4": {"excluded": [2, 4], "fraction": 0.80},
        "minus_4_9": {"excluded": [4, 9], "fraction": 0.80},
        "minus_cluster": {"excluded": [3, 5, 8], "fraction": 0.70},
    }

    seeds = [42, 123, 456]
    results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # 1. Train full model
        print(f"\n--- Full model (seed {seed}) ---")
        full_model = train_vae_fresh(config, train_ds, seed, device, label=f"full_s{seed}")
        full_latents = extract_latents(full_model, test_loader, device)
        full_pers = compute_persistence(full_latents, max_dim=1, n_samples=2000, seed=seed)
        print(f"  Full latents: {full_latents.shape}")

        for cond_name, cond_info in conditions.items():
            excluded = cond_info["excluded"]
            fraction = cond_info["fraction"]

            # 2. Random ablation (same fraction, all classes)
            n_total = len(train_ds)
            n_keep = int(n_total * fraction)
            rng = np.random.RandomState(seed + hash(cond_name) % 10000)
            rand_indices = rng.choice(n_total, n_keep, replace=False).tolist()
            rand_subset = Subset(train_ds, rand_indices)

            print(f"\n--- {cond_name}: random {fraction:.0%} ({n_keep} samples, all classes) ---")
            rand_model = train_vae_fresh(config, rand_subset, seed, device, label=f"rand_{cond_name}_s{seed}")
            rand_latents = extract_latents(rand_model, test_loader, device)
            rand_pers = compute_persistence(rand_latents, max_dim=1, n_samples=2000, seed=seed)
            w_rand = compute_wasserstein(full_pers["diagrams"], rand_pers["diagrams"])

            # 3. Categorical ablation (load cached or train)
            cat_model_path = config.models_dir / f"{config.run_name(cond_name, seed)}.pt"
            cat_model = VAE(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(device)

            if cat_model_path.exists():
                print(f"  Loading cached categorical model: {cat_model_path.name}")
                cat_model.load_state_dict(torch.load(cat_model_path, map_location=device, weights_only=True))
            else:
                print(f"  Training categorical model (excluded={excluded})")
                mask = torch.tensor([int(t) not in excluded for t in train_targets])
                cat_indices = mask.nonzero(as_tuple=True)[0].tolist()
                cat_subset = Subset(train_ds, cat_indices)
                cat_model = train_vae_fresh(config, cat_subset, seed, device, label=f"cat_{cond_name}_s{seed}")
                torch.save(cat_model.state_dict(), cat_model_path)

            cat_latents = extract_latents(cat_model, test_loader, device)
            cat_pers = compute_persistence(cat_latents, max_dim=1, n_samples=2000, seed=seed)
            w_cat = compute_wasserstein(full_pers["diagrams"], cat_pers["diagrams"])

            delta_h1 = w_cat["H1"] - w_rand["H1"]
            delta_h0 = w_cat["H0"] - w_rand["H0"]

            print(f"  Random:      W(H1)={w_rand['H1']:.2f}  W(H0)={w_rand['H0']:.2f}")
            print(f"  Categorical: W(H1)={w_cat['H1']:.2f}  W(H0)={w_cat['H0']:.2f}")
            print(f"  Delta H1:    {delta_h1:+.2f}  Delta H0: {delta_h0:+.2f}")

            results.append({
                "seed": seed,
                "condition": cond_name,
                "excluded": excluded,
                "fraction": fraction,
                "W_H1_random": float(w_rand["H1"]),
                "W_H1_categorical": float(w_cat["H1"]),
                "W_H0_random": float(w_rand["H0"]),
                "W_H0_categorical": float(w_cat["H0"]),
                "delta_H1": float(delta_h1),
                "delta_H0": float(delta_h0),
            })

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Condition':<15} {'Seed':<6} {'W(H1) rand':<12} {'W(H1) cat':<12} {'Delta H1':<12} {'Delta>0?'}")
    print(f"{'='*80}")
    n_positive = 0
    n_total_pairs = 0
    for r in results:
        pos = "YES" if r["delta_H1"] > 0 else "no"
        if r["delta_H1"] > 0:
            n_positive += 1
        n_total_pairs += 1
        print(f"{r['condition']:<15} {r['seed']:<6} {r['W_H1_random']:<12.2f} {r['W_H1_categorical']:<12.2f} {r['delta_H1']:<+12.2f} {pos}")

    print(f"\nDelta > 0 in {n_positive}/{n_total_pairs} pairs")

    # Aggregate by condition
    print(f"\n--- Aggregate by condition ---")
    for cond_name in conditions:
        deltas = [r["delta_H1"] for r in results if r["condition"] == cond_name]
        rands = [r["W_H1_random"] for r in results if r["condition"] == cond_name]
        cats = [r["W_H1_categorical"] for r in results if r["condition"] == cond_name]
        print(f"  {cond_name:<15} rand={np.mean(rands):.1f}+-{np.std(rands):.1f}  "
              f"cat={np.mean(cats):.1f}+-{np.std(cats):.1f}  "
              f"delta={np.mean(deltas):+.1f}+-{np.std(deltas):.1f}")

    total = time.time() - t0
    print(f"\nDone in {total:.0f}s")

    output = {
        "dataset": dataset_name,
        "design": "paired: same full model per seed, same persistence seed",
        "results": results,
        "summary": {
            "n_positive_delta": n_positive,
            "n_total_pairs": n_total_pairs,
        },
        "timing_s": total,
    }
    out_path = config.base_dir / "random_ablation_control.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "mnist"
    run(ds)
