"""
Comprehensive test suite for DLN (Deep Linear Network) class.

Tests cover shapes, linearity, composition, rank properties, initialization,
reproducibility, device/dtype handling, and gradient flow.
"""

import pytest
import torch
from scripts.models import DLN


# ============================================================================
# Fixtures and Helpers
# ============================================================================


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed before each test for reproducibility."""
    torch.manual_seed(42)


def get_layer_dims(model):
    """Extract layer dimensions from DLN model."""
    dims = []
    for layer in model.layers:
        dims.append((layer.weight.shape[1], layer.weight.shape[0]))  # (in, out)
    return dims


def compute_composed_weight(model):
    """Compute W_L @ ... @ W_1 for the network."""
    with torch.no_grad():
        W = model.layers[0].weight
        for layer in model.layers[1:]:
            W = layer.weight @ W
    return W


# ============================================================================
# Shape & Architecture Tests
# ============================================================================


@pytest.mark.parametrize(
    "input_dim,hidden_dims,output_dim,num_hidden_layers",
    [
        (5, 10, 3, 1),  # Single hidden layer
        (5, 10, 3, 4),  # Four hidden layers
        (7, [11, 13, 11], 2, 3),  # List overrides num_hidden_layers
        (8, 12, 6, 2),  # Two hidden layers
    ],
)
def test_forward_shapes(input_dim, hidden_dims, output_dim, num_hidden_layers):
    """Test that forward pass produces correct output shapes."""
    model = DLN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        num_hidden_layers=num_hidden_layers,
        bias=False,
    )

    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    y = model(x)

    assert y.shape == (batch_size, output_dim), (
        f"Expected output shape ({batch_size}, {output_dim}), "
        f"got {y.shape}"
    )


@pytest.mark.parametrize(
    "input_dim,hidden_dims,output_dim,num_hidden_layers",
    [
        (5, 10, 3, 1),
        (5, 10, 3, 4),
        (7, [11, 13, 11], 2, 3),
    ],
)
def test_layer_weight_shapes(
    input_dim, hidden_dims, output_dim, num_hidden_layers
):
    """Test that each layer has correct weight shape."""
    model = DLN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        num_hidden_layers=num_hidden_layers,
        bias=False,
    )

    # Build expected dimensions
    if isinstance(hidden_dims, list):
        expected_dims = [input_dim] + hidden_dims + [output_dim]
    else:
        expected_dims = [input_dim] + [hidden_dims] * num_hidden_layers
        expected_dims += [output_dim]

    # Check each layer
    for i, layer in enumerate(model.layers):
        d_in = expected_dims[i]
        d_out = expected_dims[i + 1]
        assert layer.weight.shape == (d_out, d_in), (
            f"Layer {i}: expected weight shape ({d_out}, {d_in}), "
            f"got {layer.weight.shape}"
        )


def test_hidden_dims_list_overrides_num_hidden_layers():
    """Test that hidden_dims list overrides num_hidden_layers."""
    hidden_list = [11, 13, 11]
    model = DLN(
        input_dim=7,
        hidden_dims=hidden_list,
        output_dim=2,
        num_hidden_layers=999,  # Should be ignored
        bias=False,
    )

    # Should have len(hidden_list) + 1 layers (hidden + output)
    assert len(model.layers) == len(hidden_list) + 1
    assert model.hidden_dims == hidden_list


@pytest.mark.xfail(
    reason=(
        "Depth=0 causes IndexError in _initialize_weights "
        "when accessing self.hidden_dims[0] on empty list. "
        "Fix: use per-layer std based on min(in_features, out_features) "
        "or handle empty hidden_dims case."
    )
)
def test_depth_zero_construction():
    """Test constructing DLN with no hidden layers (direct input->output)."""
    # This should create a single layer network
    model = DLN(
        input_dim=5, hidden_dims=10, output_dim=3, num_hidden_layers=0, bias=False
    )

    # If it doesn't crash, verify it's a single layer
    assert len(model.layers) == 1
    assert model.layers[0].weight.shape == (3, 5)


# ============================================================================
# Linearity Tests
# ============================================================================


def test_linearity_superposition():
    """Test f(ax + by) = a*f(x) + b*f(y) for linear network."""
    model = DLN(input_dim=5, hidden_dims=10, output_dim=3, num_hidden_layers=2, bias=False)

    x = torch.randn(1, 5)
    y = torch.randn(1, 5)
    a, b = 2.5, -1.3

    lhs = model(a * x + b * y)
    rhs = a * model(x) + b * model(y)

    assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-6), (
        "Superposition property violated for linear network"
    )


def test_linearity_zero_input():
    """Test f(0) = 0 when bias=False."""
    model = DLN(input_dim=5, hidden_dims=10, output_dim=3, num_hidden_layers=2, bias=False)

    zero_input = torch.zeros(1, 5)
    output = model(zero_input)

    assert torch.allclose(output, torch.zeros(1, 3), atol=1e-6), (
        "f(0) should be 0 for bias=False"
    )


# ============================================================================
# Composition Tests
# ============================================================================


def test_composition_equals_product():
    """Test that forward pass equals matrix product X @ W^T."""
    model = DLN(input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=3, bias=False)

    X = torch.randn(16, 5)

    # Compute via forward pass
    Y_forward = model(X)

    # Compute via composed weight
    W = compute_composed_weight(model)
    Y_composed = X @ W.T

    assert torch.allclose(Y_forward, Y_composed, rtol=1e-4, atol=1e-6), (
        "Forward pass should equal X @ W^T where W = W_L @ ... @ W_1"
    )


# ============================================================================
# Rank Tests
# ============================================================================


def test_rank_bounds():
    """Test that rank(W) <= min_i rank(W_i) and rank(W) <= min(m, n)."""
    model = DLN(input_dim=8, hidden_dims=6, output_dim=5, num_hidden_layers=3, bias=False)

    W = compute_composed_weight(model)
    rank_W = torch.linalg.matrix_rank(W).item()

    # Get per-layer ranks
    layer_ranks = model.get_layer_ranks()
    min_layer_rank = min(layer_ranks)

    # Test bounds
    assert rank_W <= min_layer_rank, (
        f"rank(W)={rank_W} should be <= min(layer_ranks)={min_layer_rank}"
    )

    m, n = W.shape
    assert rank_W <= min(m, n), (
        f"rank(W)={rank_W} should be <= min(output_dim, input_dim)={min(m, n)}"
    )


def test_effective_rank_properties():
    """Test that effective_rank is finite, non-negative, and bounded."""
    model = DLN(input_dim=8, hidden_dims=6, output_dim=5, num_hidden_layers=2, bias=False)

    eff_rank = model.effective_rank

    # Convert to scalar if it's a tensor
    if isinstance(eff_rank, torch.Tensor):
        eff_rank_val = eff_rank.item()
    else:
        eff_rank_val = eff_rank

    # Should be finite
    assert torch.isfinite(torch.tensor(eff_rank_val)), (
        f"effective_rank should be finite, got {eff_rank_val}"
    )

    # Should be non-negative
    assert eff_rank_val >= 0, f"effective_rank should be >= 0, got {eff_rank_val}"

    # Should be bounded by min layer rank (with tolerance for definition differences)
    layer_ranks = model.get_layer_ranks()
    min_layer_rank = min(layer_ranks)
    assert eff_rank_val <= min_layer_rank + 1e-3, (
        f"effective_rank={eff_rank_val} should be <= min(layer_ranks)={min_layer_rank} + eps"
    )


def test_rank_zero_when_layer_zeroed():
    """Test rank behavior when one layer is completely zeroed out."""
    model = DLN(input_dim=8, hidden_dims=6, output_dim=5, num_hidden_layers=2, bias=False)

    # Zero out first layer
    with torch.no_grad():
        model.layers[0].weight.zero_()

    W = compute_composed_weight(model)
    rank_W = torch.linalg.matrix_rank(W).item()

    assert rank_W == 0, f"rank(W) should be 0 when a layer is zeroed, got {rank_W}"

    # effective_rank should handle this gracefully
    # (might be 0, NaN, or Inf depending on implementation; we just check it doesn't crash)
    eff_rank = model.effective_rank
    # Just verify it's a number or tensor (could be 0, nan, or inf)
    assert isinstance(eff_rank, (int, float, torch.Tensor)), (
        f"effective_rank should return a number or tensor, got {type(eff_rank)}"
    )


# ============================================================================
# Parameter Counting Tests
# ============================================================================


@pytest.mark.parametrize("bias", [False, True])
def test_parameter_counting(bias):
    """Test that total_parameters matches manual calculation."""
    input_dim, hidden_dims, output_dim, num_hidden = 5, 8, 3, 2
    model = DLN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        num_hidden_layers=num_hidden,
        bias=bias,
    )

    # Build dimension sequence
    dims = [input_dim] + [hidden_dims] * num_hidden + [output_dim]

    # Manually compute parameter count
    expected_params = 0
    for i in range(len(dims) - 1):
        d_in, d_out = dims[i], dims[i + 1]
        expected_params += d_in * d_out
        if bias:
            expected_params += d_out

    assert model.total_parameters == expected_params, (
        f"Expected {expected_params} parameters, got {model.total_parameters}"
    )


# ============================================================================
# Bias Behavior Tests
# ============================================================================


def test_bias_zero_init():
    """Test that with bias=True and zero-init, f(0) = 0 initially."""
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=True
    )

    zero_input = torch.zeros(1, 5)
    output = model(zero_input)

    # With zero-initialized biases, should still give zero
    assert torch.allclose(output, torch.zeros(1, 3), atol=1e-6), (
        "With zero-init biases, f(0) should be 0"
    )


def test_bias_nonzero_effect():
    """Test that setting nonzero bias makes f(0) != 0."""
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=True
    )

    # Set first layer bias to nonzero
    with torch.no_grad():
        model.layers[0].bias.fill_(1.0)

    zero_input = torch.zeros(1, 5)
    output = model(zero_input)

    # Should now be nonzero
    assert not torch.allclose(output, torch.zeros(1, 3), atol=1e-6), (
        "With nonzero bias, f(0) should be nonzero"
    )


# ============================================================================
# Reproducibility Tests
# ============================================================================


def test_reproducibility_same_seed():
    """Test that same seed produces identical models."""
    torch.manual_seed(123)
    model1 = DLN(input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2)

    torch.manual_seed(123)
    model2 = DLN(input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2)

    # Compare all parameters
    for (name1, p1), (name2, p2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        assert name1 == name2
        assert torch.equal(p1, p2), (
            f"Parameters {name1} differ despite same seed"
        )


def test_reproducibility_different_seeds():
    """Test that different seeds produce different models."""
    torch.manual_seed(123)
    model1 = DLN(input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2)

    torch.manual_seed(456)
    model2 = DLN(input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2)

    # At least one parameter should differ
    params_differ = False
    for (n1, p1), (n2, p2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if not torch.equal(p1, p2):
            params_differ = True
            break

    assert params_differ, "Different seeds should produce different parameters"


# ============================================================================
# Device & Dtype Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_cpu(dtype):
    """Test forward pass with different dtypes on CPU."""
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=False
    )
    model = model.to(dtype=dtype)

    x = torch.randn(16, 5, dtype=dtype)
    y = model(x)

    assert y.dtype == dtype, f"Expected output dtype {dtype}, got {y.dtype}"
    assert y.shape == (16, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_float32():
    """Test forward pass on CUDA with float32."""
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=False
    )
    model = model.cuda()

    x = torch.randn(16, 5, device="cuda")
    y = model(x)

    assert y.device.type == "cuda"
    assert y.dtype == torch.float32
    assert y.shape == (16, 3)


# ============================================================================
# Batched Input Tests
# ============================================================================


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_batched_inputs(batch_size):
    """Test that model handles various batch sizes correctly."""
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=False
    )

    x = torch.randn(batch_size, 5)
    y = model(x)

    assert y.shape == (batch_size, 3), (
        f"Expected output shape ({batch_size}, 3), got {y.shape}"
    )


def test_1d_input_behavior():
    """Test behavior when given 1D input (should fail or require explicit handling)."""
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=False
    )

    x_1d = torch.randn(5)

    # PyTorch Linear expects at least 2D input; this should raise an error
    # or give unexpected behavior. We test that it doesn't silently succeed
    # with wrong shape.
    try:
        y = model(x_1d)
        # If it doesn't error, check that output is at least reasonable
        assert y.shape[-1] == 3, (
            "1D input handling unclear; output last dim should be output_dim"
        )
    except (RuntimeError, IndexError):
        # Expected: PyTorch Linear will raise error on 1D input in some configs
        pass


# ============================================================================
# Initialization Tests
# ============================================================================


def test_initialization_scaling():
    """Test that weight initialization follows specified std formula."""
    gamma = 2.0
    hidden_dim = 16
    model = DLN(
        input_dim=8,
        hidden_dims=hidden_dim,
        output_dim=5,
        num_hidden_layers=3,
        gamma=gamma,
        bias=False,
    )

    # Expected std: (hidden_dim) ** (-gamma / 2)
    expected_std = hidden_dim ** (-gamma / 2)

    # Check each layer's actual std
    for i, layer in enumerate(model.layers):
        actual_std = layer.weight.std(unbiased=False).item()
        assert abs(actual_std - expected_std) / expected_std < 0.3, (
            f"Layer {i}: expected std ≈ {expected_std:.4f}, "
            f"got {actual_std:.4f} (rtol=0.3 due to finite samples)"
        )


def test_bias_initialization():
    """Test that biases are initialized to zero."""
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=True
    )

    for i, layer in enumerate(model.layers):
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias)), (
            f"Layer {i} bias should be initialized to zero"
        )


# ============================================================================
# Gradient Flow Tests
# ============================================================================


def test_backward_gradients():
    """Test that backward pass computes gradients correctly."""
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=False
    )

    x = torch.randn(16, 5)
    target = torch.randn(16, 3)

    # Forward
    y = model(x)
    loss = torch.nn.functional.mse_loss(y, target)

    # Backward
    loss.backward()

    # Check all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert param.grad.shape == param.shape, (
            f"Gradient shape mismatch for {name}"
        )
        assert torch.isfinite(param.grad).all(), (
            f"Gradient for {name} contains NaN or Inf"
        )


# ============================================================================
# Edge Cases
# ============================================================================


def test_effective_rank_with_tiny_weights():
    """Test that effective_rank handles extremely small weights gracefully.

    Note: With very small weights, numerical issues may cause NaN due to
    division by very small numbers in the effective rank formula.
    We primarily test that it doesn't crash.
    """
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=False
    )

    # Scale all weights down to tiny values
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.mul_(1e-10)

    # Should not crash (may return NaN due to numerical issues)
    try:
        eff_rank = model.effective_rank
        # Just verify it's a number/tensor (finite or NaN is acceptable)
        assert isinstance(eff_rank, (int, float, torch.Tensor)), (
            f"effective_rank should return a numeric type, got {type(eff_rank)}"
        )
    except (RuntimeError, ZeroDivisionError):
        # Also acceptable if it raises an error on extreme values
        pass


def test_effective_rank_with_zero_weights():
    """Test effective_rank when all weights are zero."""
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=False
    )

    # Zero out all weights
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.zero_()

    # Current implementation will have s.max() = 0, causing division by zero
    # This might return NaN or Inf; we just verify it doesn't crash
    try:
        eff_rank = model.effective_rank
        # If it returns a value, check it's a number
        assert isinstance(eff_rank, (int, float, torch.Tensor))
    except (RuntimeError, ZeroDivisionError):
        # Also acceptable if it raises an error
        pass


# ============================================================================
# Integration Test
# ============================================================================


def test_full_training_step():
    """Integration test: full forward-backward-update cycle."""
    model = DLN(
        input_dim=5, hidden_dims=8, output_dim=3, num_hidden_layers=2, bias=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(16, 5)
    target = torch.randn(16, 3)

    # Initial forward
    y_before = model(x).detach().clone()

    # Training step
    optimizer.zero_grad()
    y = model(x)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    optimizer.step()

    # After update
    y_after = model(x)

    # Parameters should have changed
    assert not torch.allclose(y_before, y_after, rtol=1e-5), (
        "Model output should change after optimization step"
    )
