"""Configuration for the gap detection experiment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ExperimentConfig:
    # VAE architecture
    latent_dim: int = 16
    hidden_dim: int = 256
    channels: Tuple[int, ...] = (32, 64)

    # Training
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 128
    beta: float = 1.0  # KL weight

    # TDA
    tda_n_samples: int = 2000
    tda_max_dim: int = 1
    n_bootstrap: int = 100

    # Dataset: "mnist" or "fashion_mnist"
    dataset_name: str = "mnist"

    # Reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # Conditions: name -> list of excluded classes
    conditions: Dict[str, List[int]] = field(default_factory=lambda: {
        "full": [],
        "minus_7": [7],
        "minus_3_7": [3, 7],
        "minus_2_4": [2, 4],
        "minus_4_9": [4, 9],
        "minus_cluster": [3, 5, 8],
    })

    # Paths (standalone: output goes next to this file)
    base_dir: Path = Path(__file__).parent / "output"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def latents_dir(self) -> Path:
        return self.base_dir / "latents"

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures"

    @property
    def diagrams_dir(self) -> Path:
        return self.base_dir / "diagrams"

    def ensure_dirs(self):
        for d in [self.models_dir, self.latents_dir, self.figures_dir, self.diagrams_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run_name(self, condition: str, seed: int) -> str:
        prefix = "fmnist_" if self.dataset_name == "fashion_mnist" else ""
        return f"{prefix}{condition}_seed{seed}"
