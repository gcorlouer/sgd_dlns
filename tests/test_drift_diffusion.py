"""Unit tests for drift-diffusion tracking."""

import torch
import pytest
import math
from metrics.drift_diffusion import (
    DriftDiffusionTracker,
    param_dict_from_model,
    flatten_grads_from_param_dict,
)
from scripts.models import DLN
from scripts.teacher import Teacher, TeacherDataset


class TestUtilities:
    """Test parameter vectorization utilities."""

    def test_param_dict_from_model(self):
        """Test extracting parameters in deterministic order."""
        model = DLN(input_dim=5, hidden_dims=[7, 6], output_dim=4, gamma=1.5)
        params = param_dict_from_model(model)

        assert isinstance(params, dict)
        assert len(params) > 0
        # Should have weights for 3 layers (input->7, 7->6, 6->output)
        assert len(params) == 3

    def test_flatten_grads_from_param_dict(self):
        """Test flattening parameter gradients."""
        # Create simple tensors
        param_grads = {
            'layer1.weight': torch.randn(3, 2),
            'layer2.weight': torch.randn(4, 3),
        }

        flat = flatten_grads_from_param_dict(param_grads)

        assert flat.dim() == 1
        assert flat.shape[0] == 3*2 + 4*3  # Total parameters


class TestDriftDiffusionTracker:
    """Test DriftDiffusionTracker class."""

    @pytest.fixture
    def setup_components(self):
        """Create model, teacher, and tracker for testing."""
        torch.manual_seed(42)

        # Tiny DLN
        d_in, d_out = 5, 4
        widths = [7, 6]
        model = DLN(
            input_dim=d_in,
            hidden_dims=widths,
            output_dim=d_out,
            gamma=1.5
        )

        # Random teacher with r*=3
        rank = 3
        teacher = Teacher(
            output_dim=d_out,
            input_dim=d_in,
            rank=rank,
            max_singular_value=10.0,
            decay_rate=2.0,
            progression='linear',
            seed=42
        )

        U, S, V = teacher.components

        # Create tracker
        eta = 0.01
        batch_size = 8
        device = torch.device('cpu')

        tracker = DriftDiffusionTracker(
            model=model,
            teacher_U=U,
            teacher_V=V,
            eta=eta,
            batch_size=batch_size,
            device=device,
            mode_indices=[0, 1, 2]
        )

        return {
            'model': model,
            'teacher': teacher,
            'tracker': tracker,
            'U': U,
            'V': V,
            'rank': rank,
            'batch_size': batch_size,
            'd_in': d_in,
            'd_out': d_out,
        }

    def test_initialization(self, setup_components):
        """Test tracker initialization."""
        tracker = setup_components['tracker']
        rank = setup_components['rank']

        assert tracker.mode_indices == [0, 1, 2]
        assert tracker.batch_size == 8
        assert tracker.eta == 0.01

    def test_effective_map_shape(self, setup_components):
        """Test effective_map returns correct shape."""
        tracker = setup_components['tracker']
        d_in = setup_components['d_in']
        d_out = setup_components['d_out']

        W_eff = tracker.current_effective_map()

        assert W_eff.shape == (d_out, d_in)

    def test_per_example_grads_shape(self, setup_components):
        """Test per-example gradient computation shape."""
        model = setup_components['model']
        teacher = setup_components['teacher']
        tracker = setup_components['tracker']
        batch_size = setup_components['batch_size']

        # Generate batch
        dataset = TeacherDataset(teacher, n_samples=batch_size, seed=42)
        batch_x, batch_y = dataset.X, dataset.Y

        # Compute per-example grads
        criterion = torch.nn.MSELoss()
        per_ex_grads = tracker.compute_per_example_grads_from_batch(
            batch_x, batch_y, criterion
        )

        # Check shapes
        for name, param in model.named_parameters():
            assert name in per_ex_grads
            expected_shape = (batch_size,) + param.shape
            assert per_ex_grads[name].shape == expected_shape

    def test_update_returns_metrics(self, setup_components):
        """Test update returns expected metrics."""
        model = setup_components['model']
        teacher = setup_components['teacher']
        tracker = setup_components['tracker']
        batch_size = setup_components['batch_size']

        # Generate batch
        dataset = TeacherDataset(teacher, n_samples=batch_size, seed=42)
        batch_x, batch_y = dataset.X, dataset.Y

        # Compute per-example grads
        criterion = torch.nn.MSELoss()
        per_ex_grads = tracker.compute_per_example_grads_from_batch(
            batch_x, batch_y, criterion
        )

        # Dummy loss
        loss = torch.tensor(1.0)

        # Update
        metrics = tracker.update(loss, per_ex_grads)

        # Check global metrics exist
        assert "snr_global" in metrics
        assert "grad_norm" in metrics
        assert "sqrt_trSigma_overB" in metrics

        # Check mode metrics exist
        for j in [0, 1, 2]:
            assert f"m_mode[{j}]" in metrics
            assert f"g_mode[{j}]" in metrics
            assert f"sigma_mode[{j}]" in metrics
            assert f"snr_mode[{j}]" in metrics

    def test_snr_global_positive(self, setup_components):
        """Test that global SNR is non-negative."""
        model = setup_components['model']
        teacher = setup_components['teacher']
        tracker = setup_components['tracker']
        batch_size = setup_components['batch_size']

        # Generate batch
        dataset = TeacherDataset(teacher, n_samples=batch_size, seed=42)
        batch_x, batch_y = dataset.X, dataset.Y

        # Compute per-example grads
        criterion = torch.nn.MSELoss()
        per_ex_grads = tracker.compute_per_example_grads_from_batch(
            batch_x, batch_y, criterion
        )

        loss = torch.tensor(1.0)
        metrics = tracker.update(loss, per_ex_grads)

        assert metrics["snr_global"] >= 0

    def test_batch_size_one_returns_nan(self, setup_components):
        """Test that B=1 returns NaN for covariance-based metrics."""
        model = setup_components['model']
        teacher = setup_components['teacher']
        U = setup_components['U']
        V = setup_components['V']

        # Create tracker with B=1
        tracker = DriftDiffusionTracker(
            model=model,
            teacher_U=U,
            teacher_V=V,
            eta=0.01,
            batch_size=1,
            device=torch.device('cpu'),
            mode_indices=[0]
        )

        # Generate single sample
        dataset = TeacherDataset(teacher, n_samples=1, seed=42)
        batch_x, batch_y = dataset.X, dataset.Y

        # Compute per-example grads
        criterion = torch.nn.MSELoss()
        per_ex_grads = tracker.compute_per_example_grads_from_batch(
            batch_x, batch_y, criterion
        )

        loss = torch.tensor(1.0)
        metrics = tracker.update(loss, per_ex_grads)

        # Should return NaN for B < 2
        assert math.isnan(metrics["snr_global"])
        assert math.isnan(metrics["sqrt_trSigma_overB"])

    def test_duplicate_examples_low_variance(self, setup_components):
        """Test that duplicate examples give near-zero covariance."""
        model = setup_components['model']
        teacher = setup_components['teacher']
        tracker = setup_components['tracker']
        batch_size = setup_components['batch_size']

        # Generate single example, duplicate it
        dataset = TeacherDataset(teacher, n_samples=1, seed=42)
        single_x, single_y = dataset.X[0], dataset.Y[0]

        # Duplicate to create batch
        batch_x = single_x.unsqueeze(0).repeat(batch_size, 1)
        batch_y = single_y.unsqueeze(0).repeat(batch_size, 1)

        # Compute per-example grads
        criterion = torch.nn.MSELoss()
        per_ex_grads = tracker.compute_per_example_grads_from_batch(
            batch_x, batch_y, criterion
        )

        loss = torch.tensor(1.0)
        metrics = tracker.update(loss, per_ex_grads)

        # Covariance should be near zero (within numerical precision)
        assert metrics["sqrt_trSigma_overB"] < 1e-5

    def test_mode_coordinate_finite_difference(self, setup_components):
        """Test mode coordinate gradient via finite differences."""
        model = setup_components['model']
        tracker = setup_components['tracker']
        U = setup_components['U']
        V = setup_components['V']

        # Choose mode j=0
        j = 0
        u_j = U[:, j:j+1]
        v_j = V[:, j:j+1]

        # Compute m_j at current parameters
        W_eff = model.effective_map()
        A_j = u_j @ v_j.T
        m_j = (W_eff * A_j).sum().item()

        # Perturb a single weight slightly
        eps = 1e-5
        param_list = list(model.parameters())
        original_weight = param_list[0].data.clone()
        param_list[0].data[0, 0] += eps

        # Compute m_j after perturbation
        W_eff_perturbed = model.effective_map()
        m_j_perturbed = (W_eff_perturbed * A_j).sum().item()

        # Finite difference approximation
        fd_grad = (m_j_perturbed - m_j) / eps

        # Restore original weight
        param_list[0].data = original_weight

        # The gradient should be non-zero (rough check)
        assert abs(fd_grad) > 1e-10 or abs(m_j) < 1e-10

    def test_empty_mode_indices(self, setup_components):
        """Test that empty mode_indices skips mode computations."""
        model = setup_components['model']
        teacher = setup_components['teacher']
        U = setup_components['U']
        V = setup_components['V']

        # Create tracker with no modes
        tracker = DriftDiffusionTracker(
            model=model,
            teacher_U=U,
            teacher_V=V,
            eta=0.01,
            batch_size=8,
            device=torch.device('cpu'),
            mode_indices=[]
        )

        # Generate batch
        dataset = TeacherDataset(teacher, n_samples=8, seed=42)
        batch_x, batch_y = dataset.X, dataset.Y

        # Compute per-example grads
        criterion = torch.nn.MSELoss()
        per_ex_grads = tracker.compute_per_example_grads_from_batch(
            batch_x, batch_y, criterion
        )

        loss = torch.tensor(1.0)
        metrics = tracker.update(loss, per_ex_grads)

        # Should only have global metrics
        assert "snr_global" in metrics
        assert "grad_norm" in metrics
        assert "sqrt_trSigma_overB" in metrics

        # No mode metrics
        assert not any("m_mode" in k for k in metrics.keys())

    def test_dimension_mismatch_raises_error(self):
        """Test that mismatched U/V dimensions raise error."""
        model = DLN(input_dim=5, hidden_dims=[7, 6], output_dim=4, gamma=1.5)

        # Mismatched U and V
        U = torch.randn(4, 3)
        V = torch.randn(5, 2)  # Different rank!

        with pytest.raises(AssertionError):
            DriftDiffusionTracker(
                model=model,
                teacher_U=U,
                teacher_V=V,
                eta=0.01,
                batch_size=8,
                device=torch.device('cpu'),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
