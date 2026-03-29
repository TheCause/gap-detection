# Topological Gaps Exist Before Learning and Are Amplified by Regularization

## Reproducibility Code

This directory contains standalone Python code to reproduce all experiments from the paper.

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. MNIST and Fashion-MNIST are downloaded automatically on first run.

### Files

| File | Description |
|------|-------------|
| `config.py` | Experiment configuration (architecture, conditions, paths) |
| `utils.py` | Shared utilities (device, datasets, training, latents, centroids, Procrustes) |
| `vae.py` | Convolutional VAE model |
| `ae.py` | Deterministic autoencoder model |
| `train.py` | VAE training loop and latent extraction |
| `tda.py` | Persistent homology (Ripser) and Wasserstein distances |
| `compare.py` | Bootstrap statistical comparison |
| `visualize.py` | Figure generation |
| `run_experiment.py` | Main VAE experiment (MNIST, 6 conditions x 5 seeds) |
| `run_fashion_mnist.py` | Fashion-MNIST replication (6 conditions x 3 seeds) |
| `run_ae_experiment.py` | AE experiment with normalization |
| `pca_baseline.py` | PCA baseline (no learning) |
| `beta_sweep.py` | Beta dose-response experiment (6 beta values) |
| `ood_recon_baseline.py` | OOD reconstruction error baseline (MSE vs TDA) |
| `random_ablation_control.py` | Paired random vs categorical ablation control |
| `ghost_centroid_analysis.py` | Aspiration mechanism analysis |
| `knn_overlap.py` | k-NN overlap in pixel space |
| `knn_overlap_latent.py` | k-NN overlap in latent space |
| `mndc.py` | Mean Nearest-neighbor Distance to Complement |
| `run_gci_test.py` | GCI stress test conditions |
| `expansion_test.py` | Compensatory expansion analysis |
| `geometry_preservation.py` | Geometry preservation test |
| `normalization_check.py` | AE normalization validation |

### Pre-computed results

The `results/` directory contains JSON files with all experimental results:

| File | Description |
|------|-------------|
| `ood_recon_baseline_mnist.json` | OOD MSE baseline (MNIST, 3 seeds) |
| `ood_recon_baseline_fashion.json` | OOD MSE baseline (Fashion-MNIST, 3 seeds) |
| `fashion_mnist_tda.json` | Fashion-MNIST TDA results (3 seeds, B=100) |
| `beta_sweep.json` | Beta sweep results (6 values, B=100) |

### Running experiments

```bash
# Core VAE experiment — MNIST (~30 min on GPU, ~2h on CPU)
python run_experiment.py

# Fashion-MNIST replication (~30 min on GPU)
python run_fashion_mnist.py

# AE experiment (~30 min on GPU)
python run_ae_experiment.py

# PCA baseline (~5 min, CPU only)
python pca_baseline.py

# Beta sweep (~6h on GPU)
python beta_sweep.py

# OOD reconstruction baseline (MNIST then Fashion-MNIST)
python ood_recon_baseline.py mnist
python ood_recon_baseline.py fashion_mnist

# Ghost centroid analysis (requires trained models)
python ghost_centroid_analysis.py

# Individual analyses
python knn_overlap.py
python mndc.py
python expansion_test.py
python geometry_preservation.py
```

### Seeds

All experiments use fixed seeds {42, 123, 456, 789, 1024} for VAE/AE (5 seeds for MNIST, 3 seeds for Fashion-MNIST and OOD baseline) and random_state=42 for PCA.

### Hardware

- **MNIST main experiments**: Apple M4 (MPS backend, 32 GB)
- **Fashion-MNIST, beta sweep, OOD baseline**: NVIDIA RTX 3090 (CUDA, 24 GB)
- CPU execution is supported but slower.

### License

CC-BY-4.0
