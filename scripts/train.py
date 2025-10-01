import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
import json

from torch.utils.data import DataLoader
from tqdm import tqdm

from torch import nn
from scripts.teacher import Teacher, TeacherDataset
from scripts.models import DLN
from scripts.metrics import Observable
from scripts.plotting import (
    plot_loss_curves,
    plot_diagonal_modes,
    plot_off_diagonal_modes,
)
from pathlib import Path
import wandb


@dataclass
class ExperimentConfig:
    """Configuration for a DLN training experiment.

    Captures all hyperparameters needed for reproducibility.
    """

    # Model architecture
    input_dim: int
    hidden_dims: int | List[int]
    output_dim: int
    num_hidden_layers: int
    gamma: float
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

    # Logging configuration
    log_interval: int = 100  # How often to compute expensive metrics
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


class Trainer:
    """Trainer for Deep Linear Networks on teacher-student tasks.

    Supports configurable logging intervals, checkpoint saving, and
    comprehensive observable tracking.
    """

    def __init__(
        self,
        teacher: Teacher,
        dataset: TeacherDataset,
        model: DLN,
        lr: float = 0.01,
        batch_size: int = 1,
        num_epochs: int = 1000,
        device: str = "cpu",
        log_interval: int = 1,  # How often to log expensive metrics
        save_dir: Optional[Path] = None,  # Where to save checkpoints
        checkpoint_interval: int = 10000,  # How often to save
    ):
        """Initialize trainer.

        Args:
            teacher: Teacher matrix for generating ground truth
            dataset: Dataset of (X, Y) pairs
            model: DLN model to train
            lr: Learning rate
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            device: Device to train on (cpu/cuda)
            log_interval: Compute modes/expensive metrics every N epochs
            save_dir: Directory to save checkpoints (None = no saving)
            checkpoint_interval: Save checkpoint every N epochs
        """
        self.teacher = teacher
        self.dataset = dataset
        self.model = model.to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.checkpoint_interval = checkpoint_interval

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Training history
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []
        self.grad_norms: List[float] = []
        self.modes: List[torch.Tensor] = []
        self.logged_epochs: List[int] = []  # Track which epochs have full metrics

    @property
    def loss(self):
        """Backward compatibility property."""
        return self.loss_fn

    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            output = self.model(x)
            loss = self.loss_fn(output, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def _compute_observables(
        self, train: TeacherDataset
    ) -> Dict[str, torch.Tensor]:
        """Compute expensive observables (modes, grad norms, etc.)."""
        obs = Observable(train.teacher, self.model)
        mode_matrix = obs.mode_matrix()

        # Compute gradient norm
        grad_norm = torch.sqrt(
            sum(
                p.grad.norm() ** 2
                for p in self.model.parameters()
                if p.grad is not None and p.grad is not None
            )
        )

        return {"mode_matrix": mode_matrix, "grad_norm": grad_norm}

    def evaluate(self, test_loader: DataLoader) -> torch.Tensor:
        """Evaluate model on test set."""
        self.model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_loss += self.loss_fn(output, y).item()

        return torch.tensor(test_loss / len(test_loader))

    def save_checkpoint(self, epoch: int, path: Optional[Path] = None):
        """Save model checkpoint."""
        if path is None and self.save_dir is None:
            return

        save_path = path or (self.save_dir / f"checkpoint_epoch_{epoch}.pt")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "grad_norms": self.grad_norms,
            "modes": self.modes,
            "logged_epochs": self.logged_epochs,
        }

        torch.save(checkpoint, save_path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.test_losses = checkpoint["test_losses"]
        self.grad_norms = checkpoint.get("grad_norms", [])
        self.modes = checkpoint.get("modes", [])
        self.logged_epochs = checkpoint.get("logged_epochs", [])

        return checkpoint["epoch"]

    def train(self):
        """Main training loop with configurable logging intervals.

        This is the new, recommended training method. Computes expensive
        observables only at log_interval epochs.
        """
        train, test = self.dataset.train_test_split()

        for epoch in tqdm(range(self.num_epochs), desc="Training", unit="epoch"):
            # Create data loaders
            train_loader = DataLoader(
                train, batch_size=self.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                test, batch_size=self.batch_size, shuffle=False
            )

            # Train one epoch
            train_loss = self._train_one_epoch(train_loader)
            test_loss = self.evaluate(test_loader)

            # Store losses every epoch (cheap)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss.item())

            # Compute expensive observables at intervals
            if epoch % self.log_interval == 0 or epoch == self.num_epochs - 1:
                observables = self._compute_observables(train)
                self.modes.append(observables["mode_matrix"])
                self.grad_norms.append(observables["grad_norm"].item())
                self.logged_epochs.append(epoch)

            # Save checkpoint at intervals
            if (
                self.save_dir
                and epoch > 0
                and epoch % self.checkpoint_interval == 0
            ):
                self.save_checkpoint(epoch)

        # Final checkpoint
        if self.save_dir:
            self.save_checkpoint(self.num_epochs)

    def training_epochs(self):
        """Legacy training method for backward compatibility.

        Computes observables every epoch. Use train() for more efficient
        training with configurable log intervals.
        """
        train, test = self.dataset.train_test_split()

        for epoch in tqdm(range(self.num_epochs), desc="Training", unit="epoch"):
            train_loader = DataLoader(
                train, batch_size=self.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                test, batch_size=self.batch_size, shuffle=False
            )

            # Compute observables before training (original behavior)
            obs = Observable(train.teacher, self.model)
            mode = obs.mode_matrix()
            self.modes.append(mode)

            # Train and store losses
            train_loss = self._train_one_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.test_losses.append(self.evaluate(test_loader).item())


# Plotting functions for losses, modes etc


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Deep Linear Network on Teacher Task"
    )

    # Model architecture
    parser.add_argument("--input-dim", type=int, default=10, help="Input dimension")
    parser.add_argument(
        "--hidden-dim", type=int, default=100, help="Hidden layer width"
    )
    parser.add_argument("--output-dim", type=int, default=10, help="Output dimension")
    parser.add_argument(
        "--num-hidden-layers", type=int, default=3, help="Number of hidden layers"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.5,
        help="Initialization exponent: sigma^2 = w^(-gamma)",
    )

    # Teacher configuration
    parser.add_argument("--rank", type=int, default=4, help="Teacher matrix rank")
    parser.add_argument(
        "--max-singular-value", type=float, default=100, help="Maximum singular value"
    )
    parser.add_argument(
        "--decay-rate", type=float, default=10, help="Singular value decay rate"
    )
    parser.add_argument(
        "--progression",
        type=str,
        default="linear",
        choices=["linear", "power"],
        help="Singular value progression type",
    )

    # Dataset configuration
    parser.add_argument(
        "--n-samples", type=int, default=20, help="Number of training samples"
    )
    parser.add_argument(
        "--noise-std", type=float, default=1.0, help="Label noise standard deviation"
    )
    parser.add_argument(
        "--whiten-inputs", action="store_true", default=True, help="Whiten input data"
    )
    parser.add_argument(
        "--no-whiten-inputs",
        action="store_false",
        dest="whiten_inputs",
        help="Do not whiten input data",
    )

    # Training configuration
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-epochs", type=int, default=100000, help="Number of training epochs"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    # Experiment tracking
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Compute expensive metrics every N epochs",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (None = no saving)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save experiment configuration to JSON",
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    # Logging
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="sgd-dln", help="W&B project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity/username"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Initialize W&B if enabled
    if args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            config=vars(args),
        )
    # Create teacher
    teacher = Teacher(
        output_dim=args.output_dim,
        input_dim=args.input_dim,
        rank=args.rank,
        max_singular_value=args.max_singular_value,
        min_singular_value=1e-12,
        decay_rate=args.decay_rate,
        progression=args.progression,
        seed=args.seed,
    )

    # Generate dataset
    dataset = TeacherDataset(
        teacher,
        n_samples=args.n_samples,
        noise_std=args.noise_std,
        whiten_inputs=args.whiten_inputs,
        seed=args.seed,
    )

    # Create model
    model = DLN(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dim,
        output_dim=args.output_dim,
        num_hidden_layers=args.num_hidden_layers,
        gamma=args.gamma,
    )

    # Setup save directory if specified
    save_dir = Path(args.save_dir) if args.save_dir else None

    # Save experiment configuration
    if args.save_config and save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        config = ExperimentConfig(
            input_dim=args.input_dim,
            hidden_dims=args.hidden_dim,
            output_dim=args.output_dim,
            num_hidden_layers=args.num_hidden_layers,
            gamma=args.gamma,
            teacher_rank=args.rank,
            max_singular_value=args.max_singular_value,
            decay_rate=args.decay_rate,
            progression=args.progression,
            n_samples=args.n_samples,
            noise_std=args.noise_std,
            whiten_inputs=args.whiten_inputs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            device=args.device,
            log_interval=args.log_interval,
            save_checkpoints=(save_dir is not None),
            checkpoint_interval=args.checkpoint_interval,
            seed=args.seed,
        )
        config.save(save_dir / "config.json")

    # Create trainer with new features
    trainer = Trainer(
        teacher,
        dataset,
        model,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        log_interval=args.log_interval,
        save_dir=save_dir,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Train model (use new train() method with efficient logging)
    trainer.train()

    train_loss = trainer.train_losses
    test_loss = trainer.test_losses
    modes = trainer.modes
    logged_epochs = trainer.logged_epochs

    # Log metrics to W&B
    if args.wandb_mode != "disabled":
        for epoch, (tr_loss, te_loss) in enumerate(zip(train_loss, test_loss)):
            wandb.log({"train_loss": tr_loss, "test_loss": te_loss, "epoch": epoch})

    # Get the script's directory, then navigate to results
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Plot and save loss curves
    fname = f"iter_{len(train_loss)}_max_singular_value_{args.max_singular_value}_gamma_{args.gamma}_lr_{args.lr}_batch_{args.batch_size}_loss.png"
    fpath = results_dir / fname
    plot_loss_curves(train_loss, test_loss, save_path=fpath, show=False)
    if args.wandb_mode != "disabled":
        wandb.log({"loss_curve": wandb.Image(str(fpath))})

    # Plot diagonal modes
    fname = f"diagonal_mode_rank_{args.rank}_iter_{len(train_loss)}_max_singular_value_{args.max_singular_value}_gamma_{args.gamma}_lr_{args.lr}_batch_{args.batch_size}_loss.png"
    fpath = results_dir / fname
    plot_diagonal_modes(modes, args.rank, save_path=fpath, show=False)
    if args.wandb_mode != "disabled":
        wandb.log({"diagonal_modes": wandb.Image(str(fpath))})

    # Plot off-diagonal modes
    fname = f"off_diagonal_mode_rank_{args.rank}_iter_{len(train_loss)}_max_singular_value_{args.max_singular_value}_gamma_{args.gamma}_lr_{args.lr}_batch_{args.batch_size}_loss.png"
    fpath = results_dir / fname
    plot_off_diagonal_modes(modes, save_path=fpath, show=False)
    if args.wandb_mode != "disabled":
        wandb.log({"off_diagonal_modes": wandb.Image(str(fpath))})

    print(f"\nTraining complete! Results saved to {results_dir}")
