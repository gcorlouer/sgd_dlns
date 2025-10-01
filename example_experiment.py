"""
Example script showing how to run clean, reproducible DLN experiments.

This demonstrates the refactored training infrastructure with:
- ExperimentConfig for reproducibility
- Efficient training with log_interval
- Checkpoint saving/loading
- Observable tracking
"""

import torch
from pathlib import Path

from scripts.train import Trainer, ExperimentConfig
from scripts.teacher import Teacher, TeacherDataset
from scripts.models import DLN
from scripts.plotting import plot_loss_curves, plot_diagonal_modes


def run_experiment(save_dir: Path, seed: int = 42):
    """Run a complete experiment with config saving and checkpointing."""

    # Create experiment configuration
    config = ExperimentConfig(
        # Model architecture
        input_dim=10,
        hidden_dims=100,
        output_dim=10,
        num_hidden_layers=3,
        gamma=2.5,
        # Teacher configuration
        teacher_rank=4,
        max_singular_value=100,
        decay_rate=10,
        progression="linear",
        # Dataset configuration
        n_samples=20,
        noise_std=1.0,
        whiten_inputs=True,
        # Training configuration
        lr=1e-4,
        batch_size=1,
        num_epochs=1000,  # Reduced for quick example
        device="cpu",
        # Logging configuration
        log_interval=100,  # Compute modes every 100 epochs (10x speedup!)
        save_checkpoints=True,
        checkpoint_interval=500,
        # Reproducibility
        seed=seed,
    )

    # Save configuration
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save(save_dir / "config.json")
    print(f"Saved config to {save_dir / 'config.json'}")

    # Set seed
    if config.seed:
        torch.manual_seed(config.seed)

    # Create teacher
    teacher = Teacher(
        output_dim=config.output_dim,
        input_dim=config.input_dim,
        rank=config.teacher_rank,
        max_singular_value=config.max_singular_value,
        min_singular_value=1e-12,
        decay_rate=config.decay_rate,
        progression=config.progression,
        seed=config.seed,
    )

    # Generate dataset
    dataset = TeacherDataset(
        teacher,
        n_samples=config.n_samples,
        noise_std=config.noise_std,
        whiten_inputs=config.whiten_inputs,
        seed=config.seed,
    )

    # Create model
    model = DLN(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
        num_hidden_layers=config.num_hidden_layers,
        gamma=config.gamma,
    )

    # Create trainer with new features
    trainer = Trainer(
        teacher,
        dataset,
        model,
        lr=config.lr,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        device=config.device,
        log_interval=config.log_interval,  # Key feature!
        save_dir=save_dir,
        checkpoint_interval=config.checkpoint_interval,
    )

    print(f"\nTraining for {config.num_epochs} epochs...")
    print(f"Computing observables every {config.log_interval} epochs")
    print(f"Saving checkpoints every {config.checkpoint_interval} epochs\n")

    # Train!
    trainer.train()

    print(f"\n✓ Training complete!")
    print(f"  Train loss: {trainer.train_losses[0]:.4f} → {trainer.train_losses[-1]:.4f}")
    print(f"  Test loss: {trainer.test_losses[0]:.4f} → {trainer.test_losses[-1]:.4f}")
    print(f"  Logged {len(trainer.modes)} mode snapshots at epochs: {trainer.logged_epochs[:5]}...")

    # Plot results
    plot_loss_curves(
        trainer.train_losses,
        trainer.test_losses,
        save_path=save_dir / "loss_curves.png",
        show=False,
    )
    print(f"\nSaved plots to {save_dir}/")

    return trainer


def resume_experiment(checkpoint_path: Path, additional_epochs: int = 500):
    """Example of resuming training from a checkpoint."""

    # Load config
    config_path = checkpoint_path.parent / "config.json"
    config = ExperimentConfig.load(config_path)
    print(f"Loaded config from {config_path}")

    # Recreate teacher, dataset, model
    if config.seed:
        torch.manual_seed(config.seed)

    teacher = Teacher(
        output_dim=config.output_dim,
        input_dim=config.input_dim,
        rank=config.teacher_rank,
        max_singular_value=config.max_singular_value,
        min_singular_value=1e-12,
        decay_rate=config.decay_rate,
        progression=config.progression,
        seed=config.seed,
    )

    dataset = TeacherDataset(
        teacher,
        n_samples=config.n_samples,
        noise_std=config.noise_std,
        whiten_inputs=config.whiten_inputs,
        seed=config.seed,
    )

    model = DLN(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
        num_hidden_layers=config.num_hidden_layers,
        gamma=config.gamma,
    )

    # Create trainer
    trainer = Trainer(
        teacher,
        dataset,
        model,
        lr=config.lr,
        batch_size=config.batch_size,
        num_epochs=additional_epochs,
        device=config.device,
        log_interval=config.log_interval,
        save_dir=checkpoint_path.parent,
    )

    # Load checkpoint
    loaded_epoch = trainer.load_checkpoint(checkpoint_path)
    print(f"Resumed from epoch {loaded_epoch}")

    # Continue training
    trainer.train()

    print(f"\n✓ Continued training complete!")
    print(f"  Total epochs: {loaded_epoch + additional_epochs}")

    return trainer


if __name__ == "__main__":
    # Example 1: Run a new experiment
    print("=" * 60)
    print("Example 1: Running new experiment")
    print("=" * 60)

    experiment_dir = Path("experiments/example_run")
    trainer = run_experiment(experiment_dir, seed=42)

    print("\n" + "=" * 60)
    print("Example 2: Resume from checkpoint (optional)")
    print("=" * 60)

    # Example 2: Resume from checkpoint (uncomment to try)
    # checkpoint_path = experiment_dir / "checkpoint_epoch_500.pt"
    # if checkpoint_path.exists():
    #     trainer_resumed = resume_experiment(checkpoint_path, additional_epochs=500)

    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)
