"""
Tests for training infrastructure (Trainer, ExperimentConfig).
"""

import pytest
import torch
import json
import tempfile
from pathlib import Path

from scripts.train import Trainer, ExperimentConfig
from scripts.teacher import Teacher, TeacherDataset
from scripts.models import DLN


@pytest.fixture
def small_setup():
    """Create small teacher/dataset/model for fast testing."""
    torch.manual_seed(42)

    teacher = Teacher(
        output_dim=5, input_dim=4, rank=3, max_singular_value=10, seed=42
    )
    dataset = TeacherDataset(teacher, n_samples=20, noise_std=0.1, seed=42)
    model = DLN(
        input_dim=4, hidden_dims=6, output_dim=5, num_hidden_layers=2, gamma=2.0
    )

    return teacher, dataset, model


# ============================================================================
# ExperimentConfig Tests
# ============================================================================


def test_experiment_config_creation():
    """Test creating an ExperimentConfig."""
    config = ExperimentConfig(
        input_dim=5,
        hidden_dims=10,
        output_dim=3,
        num_hidden_layers=2,
        gamma=2.5,
        seed=42,
    )

    assert config.input_dim == 5
    assert config.hidden_dims == 10
    assert config.gamma == 2.5
    assert config.seed == 42


def test_experiment_config_save_load():
    """Test saving and loading ExperimentConfig."""
    config = ExperimentConfig(
        input_dim=7,
        hidden_dims=12,
        output_dim=4,
        num_hidden_layers=3,
        gamma=1.8,
        teacher_rank=5,
        max_singular_value=50.0,
        seed=123,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "config.json"
        config.save(save_path)

        assert save_path.exists(), "Config file should exist"

        # Load and verify
        loaded_config = ExperimentConfig.load(save_path)
        assert loaded_config.input_dim == config.input_dim
        assert loaded_config.hidden_dims == config.hidden_dims
        assert loaded_config.gamma == config.gamma
        assert loaded_config.seed == config.seed
        assert loaded_config.teacher_rank == config.teacher_rank


def test_experiment_config_to_json():
    """Test that config can be serialized to valid JSON."""
    config = ExperimentConfig(
        input_dim=5,
        hidden_dims=10,
        output_dim=3,
        num_hidden_layers=2,
        gamma=2.5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "config.json"
        config.save(save_path)

        # Load as raw JSON and verify structure
        with open(save_path, "r") as f:
            data = json.load(f)

        assert "input_dim" in data
        assert "hidden_dims" in data
        assert "gamma" in data
        assert isinstance(data, dict)


# ============================================================================
# Trainer Basic Tests
# ============================================================================


def test_trainer_initialization(small_setup):
    """Test Trainer initialization with basic parameters."""
    teacher, dataset, model = small_setup

    trainer = Trainer(
        teacher=teacher,
        dataset=dataset,
        model=model,
        lr=0.01,
        batch_size=2,
        num_epochs=10,
    )

    assert trainer.lr == 0.01
    assert trainer.batch_size == 2
    assert trainer.num_epochs == 10
    assert len(trainer.train_losses) == 0
    assert len(trainer.modes) == 0


def test_trainer_with_log_interval(small_setup):
    """Test Trainer with configurable log_interval."""
    teacher, dataset, model = small_setup

    trainer = Trainer(
        teacher=teacher,
        dataset=dataset,
        model=model,
        lr=0.01,
        batch_size=2,
        num_epochs=10,
        log_interval=3,  # Log every 3 epochs
    )

    assert trainer.log_interval == 3


def test_trainer_legacy_training_epochs(small_setup):
    """Test backward-compatible training_epochs method."""
    teacher, dataset, model = small_setup

    trainer = Trainer(
        teacher=teacher,
        dataset=dataset,
        model=model,
        lr=0.01,
        batch_size=2,
        num_epochs=5,
    )

    trainer.training_epochs()

    # Should have losses for all epochs
    assert len(trainer.train_losses) == 5
    assert len(trainer.test_losses) == 5
    # Should have modes for all epochs (legacy behavior)
    assert len(trainer.modes) == 5

    # Losses should decrease (model is learning)
    assert trainer.train_losses[-1] < trainer.train_losses[0]


def test_trainer_new_train_method(small_setup):
    """Test new train() method with log_interval."""
    teacher, dataset, model = small_setup

    trainer = Trainer(
        teacher=teacher,
        dataset=dataset,
        model=model,
        lr=0.01,
        batch_size=2,
        num_epochs=10,
        log_interval=3,  # Log every 3 epochs
    )

    trainer.train()

    # Should have losses for all epochs
    assert len(trainer.train_losses) == 10
    assert len(trainer.test_losses) == 10

    # Should have modes only at logged epochs (0, 3, 6, 9)
    expected_logged = [0, 3, 6, 9]
    assert len(trainer.modes) == len(expected_logged)
    assert trainer.logged_epochs == expected_logged

    # Should have same number of grad_norms as modes
    assert len(trainer.grad_norms) == len(trainer.modes)


def test_trainer_evaluate(small_setup):
    """Test that evaluate method works correctly."""
    teacher, dataset, model = small_setup

    trainer = Trainer(
        teacher=teacher,
        dataset=dataset,
        model=model,
        lr=0.01,
        batch_size=2,
        num_epochs=1,
    )

    train, test = dataset.train_test_split()
    test_loader = torch.utils.data.DataLoader(test, batch_size=2)

    # Evaluate before training
    initial_loss = trainer.evaluate(test_loader)
    assert isinstance(initial_loss, torch.Tensor)
    assert initial_loss.item() > 0

    # Train a bit
    train_loader = torch.utils.data.DataLoader(train, batch_size=2)
    trainer._train_one_epoch(train_loader)

    # Evaluate after training
    final_loss = trainer.evaluate(test_loader)
    assert isinstance(final_loss, torch.Tensor)
    # Loss should decrease (model is learning)
    assert final_loss.item() < initial_loss.item()


# ============================================================================
# Checkpoint Tests
# ============================================================================


def test_trainer_save_checkpoint(small_setup):
    """Test saving a checkpoint."""
    teacher, dataset, model = small_setup

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        trainer = Trainer(
            teacher=teacher,
            dataset=dataset,
            model=model,
            lr=0.01,
            batch_size=2,
            num_epochs=5,
            save_dir=save_dir,
        )

        # Train a bit
        trainer.train()

        # Should have created save_dir
        assert save_dir.exists()

        # Should have saved final checkpoint
        final_checkpoint = save_dir / "checkpoint_epoch_5.pt"
        assert final_checkpoint.exists()


def test_trainer_checkpoint_interval(small_setup):
    """Test that checkpoints are saved at correct intervals."""
    teacher, dataset, model = small_setup

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        trainer = Trainer(
            teacher=teacher,
            dataset=dataset,
            model=model,
            lr=0.01,
            batch_size=2,
            num_epochs=25,
            save_dir=save_dir,
            checkpoint_interval=10,
        )

        trainer.train()

        # Should have checkpoints at epochs 10, 20, and 25 (final)
        assert (save_dir / "checkpoint_epoch_10.pt").exists()
        assert (save_dir / "checkpoint_epoch_20.pt").exists()
        assert (save_dir / "checkpoint_epoch_25.pt").exists()


def test_trainer_load_checkpoint(small_setup):
    """Test loading a checkpoint and resuming training."""
    teacher, dataset, model = small_setup

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)

        # Train and save checkpoint
        trainer1 = Trainer(
            teacher=teacher,
            dataset=dataset,
            model=model,
            lr=0.01,
            batch_size=2,
            num_epochs=5,
            save_dir=save_dir,
        )
        trainer1.train()

        # Get final weights
        original_weights = [p.clone() for p in model.parameters()]

        # Create new model and trainer
        model2 = DLN(
            input_dim=4, hidden_dims=6, output_dim=5, num_hidden_layers=2, gamma=2.0
        )
        trainer2 = Trainer(
            teacher=teacher,
            dataset=dataset,
            model=model2,
            lr=0.01,
            batch_size=2,
            num_epochs=5,
            save_dir=save_dir,
        )

        # Load checkpoint
        epoch = trainer2.load_checkpoint(save_dir / "checkpoint_epoch_5.pt")

        assert epoch == 5
        assert len(trainer2.train_losses) == 5
        assert len(trainer2.test_losses) == 5

        # Weights should match
        for p1, p2 in zip(original_weights, model2.parameters()):
            assert torch.allclose(p1, p2)


# ============================================================================
# Observable Tracking Tests
# ============================================================================


def test_trainer_tracks_modes(small_setup):
    """Test that modes are tracked correctly."""
    teacher, dataset, model = small_setup

    trainer = Trainer(
        teacher=teacher,
        dataset=dataset,
        model=model,
        lr=0.01,
        batch_size=2,
        num_epochs=5,
        log_interval=2,
    )

    trainer.train()

    # Should have modes at epochs 0, 2, 4
    assert len(trainer.modes) == 3
    assert all(isinstance(m, torch.Tensor) for m in trainer.modes)
    assert all(m.shape == (teacher.rank, teacher.rank) for m in trainer.modes)


def test_trainer_tracks_grad_norms(small_setup):
    """Test that gradient norms are tracked."""
    teacher, dataset, model = small_setup

    trainer = Trainer(
        teacher=teacher,
        dataset=dataset,
        model=model,
        lr=0.01,
        batch_size=2,
        num_epochs=5,
        log_interval=2,
    )

    trainer.train()

    # Should have grad_norms at same epochs as modes
    assert len(trainer.grad_norms) == len(trainer.modes)
    assert all(isinstance(g, (int, float)) for g in trainer.grad_norms)
    assert all(g >= 0 for g in trainer.grad_norms)


def test_trainer_backward_compatibility_properties(small_setup):
    """Test that backward-compatible properties still work."""
    teacher, dataset, model = small_setup

    trainer = Trainer(
        teacher=teacher,
        dataset=dataset,
        model=model,
        lr=0.01,
        batch_size=2,
        num_epochs=3,
    )

    # Test loss property (backward compat)
    assert trainer.loss == trainer.loss_fn

    trainer.training_epochs()

    # Should have accumulated losses
    assert len(trainer.train_losses) == 3
    assert len(trainer.test_losses) == 3
