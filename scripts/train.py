import torch
import numpy as np

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

# wandb.init(
#    project="sgd-dln",
#    entity="geeom"   # your personal username
# )


# Code training loop class with SGD
class Trainer:
    def __init__(
        self,
        teacher: Teacher,
        dataset: TeacherDataset,
        model: DLN,
        lr: float = 0.01,
        batch_size: int = 1,
        num_epochs: int = 1000,
        device: str = "cpu",
    ):

        self.teacher = teacher
        self.dataset = dataset
        self.model = model.to(device)
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.train_losses = []
        self.test_losses = []
        self.grad_norms = []
        self.modes = []

    def online_training_loop(self):
        train, test = self.dataset.train_test_split()
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
        for x, y in tqdm(train_loader, desc="Training", unit="batch"):
            x, y = x.to(self.device), y.to(self.device)
            # Calculate and store mode
            obs = Observable(train.teacher, self.model)
            mode = obs.mode_matrix()
            # Forward
            self.model.train()
            target = self.model(x)
            train_loss = self.loss(target, y)
            self.train_losses.append(train_loss.item())
            self.test_losses.append(self.evaluate(test_loader).item())
            self.modes.append(mode)
            # Backward
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            grad_norm = torch.sqrt(
                sum(
                    [
                        p.grad.norm() ** 2
                        for p in self.model.parameters()
                        if p.grad is not None
                    ]
                )
            )
            self.grad_norms.append(grad_norm.item())

    def epoch_training_loop(self, train_loader: DataLoader):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            # Forward
            self.model.train()
            target = self.model(x)
            train_loss = self.loss(target, y)
            # backward
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            epoch_loss += train_loss.item()
        return epoch_loss / len(train_loader)

    def training_epochs(self):
        train, test = self.dataset.train_test_split()
        for i in tqdm(range(self.num_epochs), desc="Training", unit="epoch"):
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
            # Calculate and store mode
            obs = Observable(train.teacher, self.model)
            mode = obs.mode_matrix()
            self.modes.append(mode)
            # Train and Store losses
            train_loss = self.epoch_training_loop(train_loader)
            self.train_losses.append(train_loss)
            self.test_losses.append(self.evaluate(test_loader).item())
            # wandb.log({
            #            "train_loss": train_loss,
            #            "test_loss": self.evaluate(test_loader).item(),
            #            "epoch": i
            #        })

    def evaluate(self, test_loader: DataLoader):
        test_loss = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                # Forward
                target = self.model(x)
                test_loss += self.loss(target, y)
        return test_loss / len(test_loader)


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

    # Create trainer
    trainer = Trainer(
        teacher,
        dataset,
        model,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
    )
    # Train model
    trainer.training_epochs()
    train_loss = trainer.train_losses
    test_loss = trainer.test_losses
    modes = trainer.modes

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
    plot_loss_curves(train_loss, test_loss, save_path=fpath, show=True)
    if args.wandb_mode != "disabled":
        wandb.log({"loss_curve": wandb.Image(str(fpath))})

    # Plot diagonal modes
    fname = f"diagonal_mode_rank_{args.rank}_iter_{len(train_loss)}_max_singular_value_{args.max_singular_value}_gamma_{args.gamma}_lr_{args.lr}_batch_{args.batch_size}_loss.png"
    fpath = results_dir / fname
    plot_diagonal_modes(modes, args.rank, save_path=fpath, show=True)
    if args.wandb_mode != "disabled":
        wandb.log({"diagonal_modes": wandb.Image(str(fpath))})

    # Plot off-diagonal modes
    fname = f"off_diagonal_mode_rank_{args.rank}_iter_{len(train_loss)}_max_singular_value_{args.max_singular_value}_gamma_{args.gamma}_lr_{args.lr}_batch_{args.batch_size}_loss.png"
    fpath = results_dir / fname
    plot_off_diagonal_modes(modes, save_path=fpath, show=True)
    if args.wandb_mode != "disabled":
        wandb.log({"off_diagonal_modes": wandb.Image(str(fpath))})

    print(f"\nTraining complete! Results saved to {results_dir}")
