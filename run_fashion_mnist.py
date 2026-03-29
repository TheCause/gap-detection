#!/usr/bin/env python3
"""Run full Gap Detection experiment on Fashion-MNIST.

This addresses the FATAL reviewer blocker: "MNIST-only — generalize to Fashion-MNIST."
Same pipeline as run_experiment.py but with dataset_name="fashion_mnist".
Output goes to output_fashion_mnist/ to keep MNIST results separate.
"""

from pathlib import Path

from config import ExperimentConfig
from run_experiment import run


def main():
    config = ExperimentConfig(
        dataset_name="fashion_mnist",
        base_dir=Path(__file__).parent / "output_fashion_mnist",
        seeds=[42, 123, 456],
    )
    run(config)


if __name__ == "__main__":
    main()
