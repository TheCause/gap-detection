#!/bin/bash
# Run all Gap Detection experiments sequentially on GPU.
# Usage: bash run_all.sh
# Expected total time: ~4-6h on RTX 3090

set -e
PYTHON="/data/venvs/rqz/bin/python3"
cd "$(dirname "$0")"

echo "=== [1/8] MNIST VAE main experiment ==="
$PYTHON run_experiment.py

echo "=== [2/8] Fashion-MNIST VAE ==="
$PYTHON run_fashion_mnist.py

echo "=== [3/8] AE experiment ==="
$PYTHON run_ae_experiment.py

echo "=== [4/8] PCA baseline ==="
$PYTHON pca_baseline.py

echo "=== [5/8] Beta sweep ==="
$PYTHON beta_sweep.py

echo "=== [6/8] Ghost centroid analysis ==="
$PYTHON ghost_centroid_analysis.py

echo "=== [7/8] OOD reconstruction baseline (MNIST + Fashion-MNIST) ==="
$PYTHON ood_recon_baseline.py mnist
$PYTHON ood_recon_baseline.py fashion_mnist

echo "=== [8/8] Random ablation control ==="
$PYTHON random_ablation_control.py mnist

echo "=== ALL DONE ==="
