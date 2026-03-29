"""
k-NN overlap analysis on raw MNIST pixels.
Measures intrinsic inter-class proximity without any neural network.
Pure numpy implementation (no sklearn) to avoid OpenBLAS thread issues on M4.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import numpy as np
from pathlib import Path

import gzip
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_mnist_flat(data_dir="./data/MNIST/raw", train=True):
    """Load MNIST from raw IDX files as flat numpy arrays (N, 784)."""
    prefix = "train" if train else "t10k"
    img_path = os.path.join(data_dir, f"{prefix}-images-idx3-ubyte")
    lbl_path = os.path.join(data_dir, f"{prefix}-labels-idx1-ubyte")

    # Try uncompressed first, fallback to gzip
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            _, n, rows, cols = struct.unpack(">IIII", f.read(16))
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    else:
        with gzip.open(img_path + ".gz", "rb") as f:
            _, n, rows, cols = struct.unpack(">IIII", f.read(16))
            X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)

    if os.path.exists(lbl_path):
        with open(lbl_path, "rb") as f:
            _, n = struct.unpack(">II", f.read(8))
            y = np.frombuffer(f.read(), dtype=np.uint8)
    else:
        with gzip.open(lbl_path + ".gz", "rb") as f:
            _, n = struct.unpack(">II", f.read(8))
            y = np.frombuffer(f.read(), dtype=np.uint8)

    X = X.astype(np.float32) / 255.0
    return X, y


def knn_indices_batched(X, k, batch_size=1000):
    """
    Compute k nearest neighbor indices for each row of X.
    Batched to avoid allocating full 60000x60000 distance matrix.
    Returns (N, k) array of neighbor indices (excluding self).
    """
    N = X.shape[0]
    # Precompute squared norms for efficient L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    X_sq = np.sum(X ** 2, axis=1)  # (N,)

    all_indices = np.empty((N, k), dtype=np.int64)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = X[start:end]  # (B, 784)

        # Squared distances: (B, N)
        # ||batch_i - X_j||^2 = ||batch_i||^2 + ||X_j||^2 - 2 * batch_i . X_j
        batch_sq = np.sum(batch ** 2, axis=1, keepdims=True)  # (B, 1)
        dists = batch_sq + X_sq[np.newaxis, :] - 2.0 * batch @ X.T  # (B, N)

        # Set self-distance to inf
        for i in range(end - start):
            dists[i, start + i] = np.inf

        # argpartition is O(N) vs O(N log N) for argsort
        idx = np.argpartition(dists, k, axis=1)[:, :k]

        # Sort the k nearest by actual distance (for consistency)
        for i in range(end - start):
            sub_dists = dists[i, idx[i]]
            order = np.argsort(sub_dists)
            idx[i] = idx[i][order]

        all_indices[start:end] = idx

        if start % 5000 == 0:
            print(f"    k-NN batch {start}/{N}...")

    return all_indices


def compute_overlap_matrix(X, y, k=50):
    """
    For each sample, find k nearest neighbors.
    Return 10x10 matrix where M[i][j] = mean fraction of
    class-i samples whose k-NN belong to class j.
    """
    indices = knn_indices_batched(X, k)

    classes = np.arange(10)
    overlap = np.zeros((10, 10))

    for c in classes:
        mask_c = y == c
        neighbor_labels = y[indices[mask_c]]  # (n_c, k)
        for j in classes:
            overlap[c, j] = np.mean(neighbor_labels == j)

    return overlap


def run_knn_analysis(k_values=None, output_dir=None):
    if k_values is None:
        k_values = [10, 30, 50, 100, 200]
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent.parent / "output" / "experiment"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST training set...")
    X, y = load_mnist_flat(train=True)
    print(f"  Shape: {X.shape}, classes: {np.bincount(y)}")

    results = {}
    for k in k_values:
        t0 = time.time()
        print(f"\nComputing k-NN overlap (k={k})...")
        M = compute_overlap_matrix(X, y, k=k)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s")

        results[k] = {
            "matrix": M.tolist(),
            "time_s": round(dt, 2)
        }

        # Print key pairs
        print(f"  overlap(3,7) = {M[3,7]:.4f}")
        print(f"  overlap(7,3) = {M[7,3]:.4f}")
        print(f"  overlap(3,5) = {M[3,5]:.4f}")
        print(f"  overlap(5,8) = {M[5,8]:.4f}")
        print(f"  overlap(4,9) = {M[4,9]:.4f}")

    # Primary analysis at k=50
    M50 = np.array(results[50]["matrix"])

    # Rank all pairs by overlap
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            sym = (M50[i, j] + M50[j, i]) / 2
            pairs.append((i, j, sym))
    pairs.sort(key=lambda x: -x[2])

    print("\n=== TOP 10 MOST OVERLAPPING PAIRS (k=50) ===")
    for rank, (i, j, v) in enumerate(pairs[:10], 1):
        print(f"  {rank}. ({i},{j}): {v:.4f}")

    # Stability check across k
    print("\n=== STABILITY CHECK: overlap(3,7) across k ===")
    for k in k_values:
        M = np.array(results[k]["matrix"])
        sym = (M[3, 7] + M[7, 3]) / 2
        print(f"  k={k:>3d}: {sym:.4f}")

    # Ranking stability
    print("\n=== RANKING STABILITY (top-5 pairs per k) ===")
    for k in k_values:
        M = np.array(results[k]["matrix"])
        p = []
        for i in range(10):
            for j in range(i + 1, 10):
                p.append((i, j, (M[i, j] + M[j, i]) / 2))
        p.sort(key=lambda x: -x[2])
        top5 = [(i, j) for i, j, _ in p[:5]]
        print(f"  k={k:>3d}: {top5}")

    # Compute relevance to experiment conditions
    print("\n=== RELEVANCE TO EXPERIMENT CONDITIONS ===")
    conditions = {
        "minus_7": [7],
        "minus_3_7": [3, 7],
        "minus_cluster": [3, 5, 8],
    }
    for cond, excluded in conditions.items():
        remaining = [c for c in range(10) if c not in excluded]
        overlaps = []
        for ex in excluded:
            for rem in remaining:
                overlaps.append((M50[ex, rem] + M50[rem, ex]) / 2)
        mean_ov = np.mean(overlaps)
        internal = []
        for i, ex1 in enumerate(excluded):
            for ex2 in excluded[i+1:]:
                internal.append((M50[ex1, ex2] + M50[ex2, ex1]) / 2)
        int_ov = np.mean(internal) if internal else 0
        print(f"  {cond}: excluded<->remaining={mean_ov:.4f}, internal={int_ov:.4f}")

    # Save results
    save_data = {
        "k_values": k_values,
        "primary_k": 50,
        "results": {str(k): results[k] for k in k_values},
        "top_pairs_k50": [(i, j, round(v, 6)) for i, j, v in pairs],
        "conditions_analysis": {}
    }
    for cond, excluded in conditions.items():
        remaining = [c for c in range(10) if c not in excluded]
        overlaps = []
        for ex in excluded:
            for rem in remaining:
                overlaps.append((M50[ex, rem] + M50[rem, ex]) / 2)
        internal = []
        for i, ex1 in enumerate(excluded):
            for ex2 in excluded[i+1:]:
                internal.append((M50[ex1, ex2] + M50[ex2, ex1]) / 2)
        save_data["conditions_analysis"][cond] = {
            "excluded": excluded,
            "mean_overlap_with_remaining": round(float(np.mean(overlaps)), 6),
            "internal_overlap": round(float(np.mean(internal)), 6) if internal else 0,
        }

    with open(output_dir / "knn_overlap.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_dir / 'knn_overlap.json'}")

    # Plot heatmap for k=50
    fig, ax = plt.subplots(figsize=(8, 7))
    M_plot = M50.copy()
    np.fill_diagonal(M_plot, np.nan)

    im = ax.imshow(M_plot, cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Neighbor class")
    ax.set_ylabel("Source class")
    ax.set_title("k-NN Overlap Matrix (k=50, MNIST pixels, diagonal masked)")

    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            val = M_plot[i, j]
            color = "white" if val > 0.04 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Fraction of k-NN")
    plt.tight_layout()
    fig_path = output_dir / "figures" / "knn_overlap_matrix.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved to {fig_path}")

    return save_data


if __name__ == "__main__":
    run_knn_analysis()
