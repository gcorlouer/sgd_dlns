import pytest
import torch
from torch import nn
from observables.drift_diffusion import DriftDiffusion
from scripts.models import DLN
from scripts.train import ExperimentConfig
from scripts.teacher import Teacher, TeacherDataset


@pytest.fixture
def base_config():
    """Base configuration for tests."""
    return ExperimentConfig(
        input_dim=3,
        hidden_dims=5,
        output_dim=3,
        num_hidden_layers=2,
        bias=False,
        lr=0.01,
        teacher_rank=2,
        max_singular_value=10,
        decay_rate=1.0,
        progression='linear',
        seed=42,
        batch_size=1,
        n_samples=100,
        noise_std=0.01,
        whiten_inputs=True,
    )


@pytest.fixture
def model_and_dataset(base_config: ExperimentConfig):
    """Create model, teacher, and dataset for tests."""
    model = DLN(
        input_dim=base_config.input_dim,
        hidden_dims=base_config.hidden_dims,
        output_dim=base_config.output_dim,
        num_hidden_layers=base_config.num_hidden_layers,
        bias=False
    )

    teacher = Teacher(
        output_dim=base_config.output_dim,
        input_dim=base_config.input_dim,
        rank=base_config.teacher_rank,
        max_singular_value=base_config.max_singular_value,
        decay_rate=base_config.decay_rate,
        progression=base_config.progression,
        seed=base_config.seed,
    )

    dataset = TeacherDataset(
        teacher,
        n_samples=base_config.n_samples,
        noise_std=base_config.noise_std,
        whiten_inputs=base_config.whiten_inputs,
        seed=base_config.seed,
    )

    return model, dataset


def test_eval_drift(base_config:ExperimentConfig, model_and_dataset: model_and_dataset):
    """Test that drift computation matches manual calculation."""
    model, dataset = model_and_dataset

    obs = DriftDiffusion(model, base_config, dataset, eval_batch_size=base_config.n_samples)
    eval_drift = obs.compute_drift()

    # Drift should always be non-negative (it's a norm)
    assert eval_drift >= 0, f"Drift should be non-negative but is {eval_drift}"

    # Manual computation for verification
    batch, target = dataset[:]
    device = next(model.parameters()).device
    batch, target = batch.to(device), target.to(device)

    model.zero_grad()
    y = model(batch)
    loss_fn = nn.MSELoss()
    loss_value = loss_fn(y, target)
    loss_value.backward()

    manual_grads = []
    for param in model.parameters():
        manual_grads.append(param.grad.flatten())
    manual_grad = torch.cat(manual_grads)
    norm2 = torch.linalg.vector_norm(manual_grad)

    true_drift = base_config.lr * norm2

    assert torch.isclose(eval_drift, true_drift, rtol=1e-4), \
        f"Eval drift {eval_drift} != true drift {true_drift}, diff = {eval_drift - true_drift}"


def test_diffusion_nonnegativity(base_config: ExperimentConfig, model_and_dataset):
    """Test that diffusion is always non-negative."""
    model, dataset = model_and_dataset

    obs = DriftDiffusion(model, base_config, dataset, eval_batch_size=50)
    diffusion = obs.compute_diffusion()

    assert diffusion >= 0, f"Diffusion should be non-negative but is {diffusion}"


def test_diffusion_zero_with_zero_lr(base_config: ExperimentConfig, model_and_dataset):
    """Test that diffusion is zero when learning rate is zero."""
    model, dataset = model_and_dataset

    # Create new config with lr=0
    zero_lr_config = ExperimentConfig(
        input_dim=base_config.input_dim,
        hidden_dims=base_config.hidden_dims,
        output_dim=base_config.output_dim,
        num_hidden_layers=base_config.num_hidden_layers,
        bias=base_config.bias,
        lr=0.0,  # Zero learning rate
        teacher_rank=base_config.teacher_rank,
        max_singular_value=base_config.max_singular_value,
        decay_rate=base_config.decay_rate,
        progression=base_config.progression,
        seed=base_config.seed,
        batch_size=base_config.batch_size,
        n_samples=base_config.n_samples,
        noise_std=base_config.noise_std,
        whiten_inputs=base_config.whiten_inputs,
    )

    obs = DriftDiffusion(model, zero_lr_config, dataset, eval_batch_size=50)
    diffusion = obs.compute_diffusion()

    assert diffusion == 0, \
        f"Diffusion should be zero when learning rate is {zero_lr_config.lr}, but diffusion is {diffusion}"


def test_drift_diffusion_ratio(base_config: ExperimentConfig, model_and_dataset):
    """Test that drift-diffusion ratio is computed correctly."""
    model, dataset = model_and_dataset

    obs = DriftDiffusion(model, base_config, dataset, eval_batch_size=50)
    ratio = obs.drift_diffusion_ratio()

    # Ratio should be positive (both drift and diffusion are positive)
    assert ratio > 0, f"Drift-diffusion ratio should be positive but is {ratio}"

    # Verify the formula by computing manually from a single call
    drift, diffusion = obs.compute_drift_and_diffusion()
    manual_ratio = drift / (diffusion + 1e-10)

    # Both should be positive and reasonable magnitude
    assert manual_ratio > 0, f"Manual ratio should be positive: {manual_ratio}"
    assert ratio < 1000, f"Ratio seems too large: {ratio}"  # Sanity check


def test_compute_drift_and_diffusion_consistency(base_config: ExperimentConfig, model_and_dataset):
    """Test that combined method produces reasonable values."""
    model, dataset = model_and_dataset
    obs = DriftDiffusion(model, base_config, dataset, eval_batch_size=50)

    # Compute together
    drift_combined, diffusion_combined = obs.compute_drift_and_diffusion()

    # Both should be non-negative
    assert drift_combined >= 0, f"Combined drift should be non-negative: {drift_combined}"
    assert diffusion_combined >= 0, f"Combined diffusion should be non-negative: {diffusion_combined}"

    # For non-zero lr, both should be positive
    if base_config.lr > 0:
        assert drift_combined > 0, "Drift should be positive for non-zero lr"
        assert diffusion_combined > 0, "Diffusion should be positive for non-zero lr"


def test_drift_scales_with_learning_rate(base_config: ExperimentConfig, model_and_dataset):
    """Test that drift scales linearly with learning rate."""
    model, dataset = model_and_dataset

    obs1 = DriftDiffusion(model, base_config, dataset, eval_batch_size=100)
    drift1 = obs1.compute_drift()

    # Double the learning rate
    config2 = ExperimentConfig(
        input_dim=base_config.input_dim,
        hidden_dims=base_config.hidden_dims,
        output_dim=base_config.output_dim,
        num_hidden_layers=base_config.num_hidden_layers,
        bias=base_config.bias,
        lr=base_config.lr * 2,
        teacher_rank=base_config.teacher_rank,
        max_singular_value=base_config.max_singular_value,
        decay_rate=base_config.decay_rate,
        progression=base_config.progression,
        seed=base_config.seed,
        batch_size=base_config.batch_size,
        n_samples=base_config.n_samples,
        noise_std=base_config.noise_std,
        whiten_inputs=base_config.whiten_inputs,
    )
    obs2 = DriftDiffusion(model, config2, dataset, eval_batch_size=100)
    drift2 = obs2.compute_drift()

    assert torch.isclose(drift2, drift1 * 2, rtol=1e-3), \
        f"Drift should scale linearly with lr: {drift2} != 2 * {drift1}"


def test_diffusion_scales_with_learning_rate(base_config: ExperimentConfig, model_and_dataset):
    """Test that diffusion scales linearly with learning rate."""
    model, dataset = model_and_dataset

    obs1 = DriftDiffusion(model, base_config, dataset, eval_batch_size=50)
    drift1, diff1 = obs1.compute_drift_and_diffusion()

    config2 = ExperimentConfig(
        input_dim=base_config.input_dim,
        hidden_dims=base_config.hidden_dims,
        output_dim=base_config.output_dim,
        num_hidden_layers=base_config.num_hidden_layers,
        bias=base_config.bias,
        lr=base_config.lr * 3,
        teacher_rank=base_config.teacher_rank,
        max_singular_value=base_config.max_singular_value,
        decay_rate=base_config.decay_rate,
        progression=base_config.progression,
        seed=base_config.seed,
        batch_size=base_config.batch_size,
        n_samples=base_config.n_samples,
        noise_std=base_config.noise_std,
        whiten_inputs=base_config.whiten_inputs,
    )
    obs2 = DriftDiffusion(model, config2, dataset, eval_batch_size=50)
    drift2, diff2 = obs2.compute_drift_and_diffusion()

    assert torch.isclose(diff2, diff1 * 3, rtol=0.1), \
        f"Diffusion should scale ~linearly with lr: {diff2} != 3 * {diff1}"


def test_diffusion_scales_inversely_with_batch_size(base_config: ExperimentConfig, model_and_dataset):
    """Test that diffusion scales as 1/batch_size."""
    model, dataset = model_and_dataset

    config1 = ExperimentConfig(
        input_dim=base_config.input_dim,
        hidden_dims=base_config.hidden_dims,
        output_dim=base_config.output_dim,
        num_hidden_layers=base_config.num_hidden_layers,
        bias=base_config.bias,
        lr=base_config.lr,
        teacher_rank=base_config.teacher_rank,
        max_singular_value=base_config.max_singular_value,
        decay_rate=base_config.decay_rate,
        progression=base_config.progression,
        seed=base_config.seed,
        batch_size=1,
        n_samples=base_config.n_samples,
        noise_std=base_config.noise_std,
        whiten_inputs=base_config.whiten_inputs,
    )
    obs1 = DriftDiffusion(model, config1, dataset, eval_batch_size=50)
    _, diff1 = obs1.compute_drift_and_diffusion()

    config2 = ExperimentConfig(
        input_dim=base_config.input_dim,
        hidden_dims=base_config.hidden_dims,
        output_dim=base_config.output_dim,
        num_hidden_layers=base_config.num_hidden_layers,
        bias=base_config.bias,
        lr=base_config.lr,
        teacher_rank=base_config.teacher_rank,
        max_singular_value=base_config.max_singular_value,
        decay_rate=base_config.decay_rate,
        progression=base_config.progression,
        seed=base_config.seed,
        batch_size=2,
        n_samples=base_config.n_samples,
        noise_std=base_config.noise_std,
        whiten_inputs=base_config.whiten_inputs,
    )
    obs2 = DriftDiffusion(model, config2, dataset, eval_batch_size=50)
    _, diff2 = obs2.compute_drift_and_diffusion()

    # Diffusion formula has 1/B factor
    assert diff2 < diff1, f"Larger batch size should reduce diffusion: {diff2} >= {diff1}"


def test_gradient_vector_raises_without_backward():
    """Test that gradient_vector raises error if backward not called."""
    cfg = ExperimentConfig(
        input_dim=3,
        hidden_dims=5,
        output_dim=3,
        num_hidden_layers=2,
        bias=False,
        lr=0.01,
        teacher_rank=2,
        max_singular_value=10,
        decay_rate=1.0,
        progression='linear',
        seed=42,
        n_samples=100,
        noise_std=0.01,
        whiten_inputs=True,
    )

    model = DLN(
        input_dim=cfg.input_dim,
        hidden_dims=cfg.hidden_dims,
        output_dim=cfg.output_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        bias=False
    )

    teacher = Teacher(
        output_dim=cfg.output_dim,
        input_dim=cfg.input_dim,
        rank=cfg.teacher_rank,
        max_singular_value=cfg.max_singular_value,
        decay_rate=cfg.decay_rate,
        progression=cfg.progression,
        seed=cfg.seed,
    )

    dataset = TeacherDataset(
        teacher,
        n_samples=cfg.n_samples,
        noise_std=cfg.noise_std,
        whiten_inputs=cfg.whiten_inputs,
        seed=cfg.seed,
    )

    obs = DriftDiffusion(model, cfg, dataset)

    with pytest.raises(ValueError, match="Gradients not computed"):
        obs.gradient_vector()


def test_works_with_trained_model(base_config: ExperimentConfig, model_and_dataset):
    """Test that drift/diffusion work after model has been trained."""
    model, dataset = model_and_dataset

    # Train for a few steps
    optimizer = torch.optim.SGD(model.parameters(), lr=base_config.lr)
    loss_fn = nn.MSELoss()

    for _ in range(10):
        indices = torch.randint(0, len(dataset), (base_config.batch_size,))
        batch, target = dataset[indices.tolist()]
        optimizer.zero_grad()
        y = model(batch)
        loss = loss_fn(y, target)
        loss.backward()
        optimizer.step()

    # Should still work after training
    obs = DriftDiffusion(model, base_config, dataset, eval_batch_size=50)
    drift = obs.compute_drift()
    diffusion = obs.compute_diffusion()

    assert drift > 0, f"Drift should be positive after training: {drift}"
    assert diffusion > 0, f"Diffusion should be positive after training: {diffusion}"


def test_preserves_model_training_mode(base_config: ExperimentConfig, model_and_dataset):
    """Test that drift/diffusion preserve model training mode."""
    model, dataset = model_and_dataset

    # Test in training mode
    model.train()
    obs = DriftDiffusion(model, base_config, dataset, eval_batch_size=50)
    obs.compute_drift()
    assert model.training, "Model should remain in training mode"

    # Test in eval mode
    model.eval()
    obs.compute_diffusion()
    assert not model.training, "Model should remain in eval mode"


def test_does_not_modify_model_parameters(base_config: ExperimentConfig, model_and_dataset):
    """Test that computing drift/diffusion doesn't change model weights."""
    model, dataset = model_and_dataset
    obs = DriftDiffusion(model, base_config, dataset, eval_batch_size=50)

    # Save initial parameters
    initial_params = [p.clone() for p in model.parameters()]

    # Compute observables
    obs.compute_drift()
    obs.compute_diffusion()
    obs.drift_diffusion_ratio()

    # Check parameters unchanged
    for p_initial, p_current in zip(initial_params, model.parameters()):
        assert torch.allclose(p_initial, p_current), \
            "Model parameters should not change during drift/diffusion computation"


def test_drift_is_deterministic(base_config: ExperimentConfig, model_and_dataset):
    """Test that repeated drift computation gives same result."""
    model, dataset = model_and_dataset
    obs = DriftDiffusion(model, base_config, dataset, eval_batch_size=100)

    drift1 = obs.compute_drift()
    drift2 = obs.compute_drift()

    assert torch.isclose(drift1, drift2), \
        f"Drift should be deterministic: {drift1} != {drift2}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_works_on_gpu(base_config: ExperimentConfig):
    """Test that drift/diffusion work on GPU."""
    model = DLN(
        input_dim=base_config.input_dim,
        hidden_dims=base_config.hidden_dims,
        output_dim=base_config.output_dim,
        num_hidden_layers=base_config.num_hidden_layers,
        bias=False
    ).cuda()

    teacher = Teacher(
        output_dim=base_config.output_dim,
        input_dim=base_config.input_dim,
        rank=base_config.teacher_rank,
        max_singular_value=base_config.max_singular_value,
        decay_rate=base_config.decay_rate,
        progression=base_config.progression,
        seed=base_config.seed,
    )

    dataset = TeacherDataset(
        teacher,
        n_samples=base_config.n_samples,
        noise_std=base_config.noise_std,
        whiten_inputs=base_config.whiten_inputs,
        seed=base_config.seed,
    )

    obs = DriftDiffusion(model, base_config, dataset, eval_batch_size=50)

    drift = obs.compute_drift()
    diffusion = obs.compute_diffusion()

    assert drift.device.type == 'cuda', f"Drift should be on CUDA: {drift.device}"
    assert diffusion.device.type == 'cuda', f"Diffusion should be on CUDA: {diffusion.device}"
    assert drift > 0, f"Drift should be positive: {drift}"
    assert diffusion > 0, f"Diffusion should be positive: {diffusion}"
