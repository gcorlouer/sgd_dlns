"""Configuration for DLN experiments."""
import json
from dataclasses import dataclass, asdict
from typing import Optional, Union, List
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for a DLN training experiment.

    Captures all hyperparameters needed for reproducibility.
    """

    # Model architecture
    input_dim: int = 4
    hidden_dims: Union[int, List[int]] = 10
    output_dim: int = 4
    num_hidden_layers: int = 3
    gamma: float = 2
    bias: bool = False

    # Teacher configuration
    teacher_rank: int = 4
    max_singular_value: float = 100.0
    decay_rate: float = 10.0
    progression: str = "linear"

    # Dataset configuration
    n_samples: int = 20
    noise_std: float = 1.0
    whiten_inputs: bool = True

    # Training configuration
    lr: float = 1e-4
    batch_size: int = 1
    num_epochs: int = 100000
    device: str = "cpu"

    # Observables
    eval_batch_size: int = 1024  # Number of samples to evaluate empirical gradient

    # Logging configuration
    log_interval: int = 100  # How often to compute expensive metrics
    drift_diffusion_interval: int = 100
    save_dir: Optional[str] = None
    save_checkpoints: bool = False
    checkpoint_interval: int = 10000

    # Reproducibility
    seed: Optional[int] = None

    def save(self, path: Path):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls(**json.load(f))
