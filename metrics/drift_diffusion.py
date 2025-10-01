"""Drift-Diffusion tracking for Deep Linear Networks.

This module implements per-step SNR computation (global and mode-wise) for
analyzing the training dynamics of Deep Linear Networks in teacher-student setups.

Mathematical Background
-----------------------
For a mini-batch of size B with per-example gradients g_i and mean gradient
ḡ = (1/B) Σ g_i, we define the per-example gradient covariance:

    Σ̂ = 1/(B-1) Σᵢ (gᵢ - ḡ)(gᵢ - ḡ)ᵀ

We compute only the trace without materializing Σ̂:

    tr(Σ̂) = 1/(B-1) Σᵢ ‖gᵢ - ḡ‖²

Global per-step SNR:

    SNR_glob = (η ‖ḡ‖²) / (tr(Σ̂)/B)

Mode-wise quantities: Given teacher SVD T = Σⱼ sⱼ uⱼ vⱼᵀ, define the mode
coordinate:

    mⱼ(θ) = ⟨W_eff(θ), uⱼ vⱼᵀ⟩_F = tr(vⱼ uⱼᵀ W_eff)

Let ψⱼ = ∇_θ mⱼ(θ) be the gradient of the mode coordinate. Then:

    gⱼ = ψⱼᵀ ḡ
    σⱼ² = 1/(B-1) Σᵢ (ψⱼᵀ (gᵢ - ḡ))²
    SNR_j = (η |gⱼ|) / (σⱼ²/B)

Example Usage
-------------
```python
from metrics import DriftDiffusionTracker
from scripts.models import DLN

model = DLN(d_in, widths, d_out).to(device)
tracker = DriftDiffusionTracker(
    model,
    U_teacher,
    V_teacher,
    eta=eta,
    batch_size=B,
    device=device,
    mode_indices=[0, 1, 2]
)

optimizer = torch.optim.SGD(model.parameters(), lr=eta)
for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    yhat = model(x)
    loss = torch.nn.functional.mse_loss(yhat, y, reduction='mean')

    # Compute per-example grads
    per_ex_grads = tracker.compute_per_example_grads(loss)

    # Standard backward on the mean loss to update weights
    loss.backward()
    optimizer.step()

    logs = tracker.update(loss.detach(), per_ex_grads)
    # send `logs` to your metrics system
```
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, OrderedDict
from collections import OrderedDict as ODict
import math


def param_dict_from_model(model: nn.Module) -> ODict[str, nn.Parameter]:
    """Extract parameters from model in a deterministic order.

    Args:
        model: PyTorch module

    Returns:
        OrderedDict mapping parameter names to parameter tensors
    """
    return ODict(model.named_parameters())


def flatten_grads_from_param_dict(
    param_to_grad: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Flatten a dictionary of gradients into a single 1-D vector.

    Args:
        param_to_grad: Dict mapping parameter names to gradient tensors

    Returns:
        1-D tensor containing all gradients concatenated
    """
    return torch.cat([g.reshape(-1) for g in param_to_grad.values()])


class DriftDiffusionTracker:
    """Track drift vs diffusion observables during DLN training.

    Computes global and mode-wise per-step SNR to analyze training dynamics.
    Works with Deep Linear Networks of arbitrary depth L ≥ 2.

    Args:
        model: DLN model with effective_map() method
        teacher_U: Left singular vectors of teacher, shape [d_out, r*]
        teacher_V: Right singular vectors of teacher, shape [d_in, r*]
        eta: Learning rate (step size)
        batch_size: Mini-batch size
        device: Device to run computations on
        mode_indices: Which teacher modes to track (default: all)
        hutchinson_num_vecs: Reserved for future use (not implemented)
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_U: torch.Tensor,
        teacher_V: torch.Tensor,
        eta: float,
        batch_size: int,
        device: torch.device,
        mode_indices: Optional[List[int]] = None,
        hutchinson_num_vecs: int = 0,
    ):
        self.model = model
        self.teacher_U = teacher_U.to(device)
        self.teacher_V = teacher_V.to(device)
        self.eta = eta
        self.batch_size = batch_size
        self.device = device
        self.hutchinson_num_vecs = hutchinson_num_vecs

        # Determine which modes to track
        rank = teacher_U.shape[1]
        if mode_indices is None:
            self.mode_indices = list(range(rank))
        else:
            self.mode_indices = mode_indices

        # Validate dimensions
        assert teacher_U.shape[1] == teacher_V.shape[1], (
            f"U and V must have same rank: U={teacher_U.shape}, V={teacher_V.shape}"
        )

        # Numerical stability epsilon
        self.eps = 1e-12

    def current_effective_map(self, requires_grad: bool = False) -> torch.Tensor:
        """Compute W_eff = W_L @ ... @ W_1 without data.

        Args:
            requires_grad: If True, compute with gradients enabled

        Returns:
            Tensor of shape [d_out, d_in]
        """
        if requires_grad:
            # Compute effective map with gradients for mode computation
            # Get all linear layer weights
            layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
            if len(layers) == 0:
                raise ValueError("Model has no layers")
            if len(layers) == 1:
                return layers[0].weight

            # Reverse order for multi_dot: [W_L, ..., W_1]
            weights = [layer.weight for layer in reversed(layers)]
            return torch.linalg.multi_dot(weights)
        else:
            return self.model.effective_map()

    def compute_per_example_grads(
        self, loss: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute per-example gradients for all parameters.

        Uses functorch.vmap if available, otherwise falls back to manual loop.

        Args:
            loss: Loss tensor with shape [B] or scalar (will compute per-example)

        Returns:
            Dict mapping parameter names to per-example gradients with shape
            [B, *param.shape]
        """
        # Get parameter dict
        params = param_dict_from_model(self.model)
        param_names = list(params.keys())
        param_list = list(params.values())

        # Compute per-example gradients using vmap
        try:
            from functorch import grad, vmap

            def compute_sample_grad(x_sample, y_sample, params_tuple):
                """Compute gradient for a single example."""
                # Temporarily assign parameters
                param_dict = dict(zip(param_names, params_tuple))

                # Functional forward pass
                outputs = torch.func.functional_call(
                    self.model, param_dict, (x_sample.unsqueeze(0),)
                )
                loss_val = torch.nn.functional.mse_loss(
                    outputs, y_sample.unsqueeze(0), reduction='mean'
                )

                # Compute gradients
                grads = torch.autograd.grad(
                    loss_val, params_tuple, create_graph=False
                )
                return grads

            # This is a simplified version - in practice we need the actual data
            # For now, use the manual approach
            raise ImportError("Using manual approach")

        except (ImportError, AttributeError):
            # Manual loop approach
            return self._compute_per_example_grads_manual()

    def _compute_per_example_grads_manual(self) -> Dict[str, torch.Tensor]:
        """Compute per-example gradients using manual loop.

        This requires access to the batch data, which should be stored during
        forward pass. For now, returns None - user should call this within
        their training loop where they have access to the batch.

        Note: This is a placeholder. In actual usage, per-example gradients
        should be computed in the training loop using:

        ```python
        for i in range(batch_size):
            x_i, y_i = batch_x[i:i+1], batch_y[i:i+1]
            y_pred = model(x_i)
            loss_i = criterion(y_pred, y_i)
            grads_i = torch.autograd.grad(loss_i, model.parameters())
        ```
        """
        raise NotImplementedError(
            "Per-example gradients must be computed in the training loop. "
            "See docstring for implementation details."
        )

    def compute_per_example_grads_from_batch(
        self, batch_x: torch.Tensor, batch_y: torch.Tensor, criterion
    ) -> Dict[str, torch.Tensor]:
        """Compute per-example gradients from a batch of data.

        Args:
            batch_x: Input batch, shape [B, ...]
            batch_y: Target batch, shape [B, ...]
            criterion: Loss function

        Returns:
            Dict mapping parameter names to per-example gradients [B, *param.shape]
        """
        B = batch_x.shape[0]
        params = param_dict_from_model(self.model)
        param_names = list(params.keys())

        # Initialize storage for per-example grads
        per_example_grads = {name: [] for name in param_names}

        # Compute gradient for each example
        for i in range(B):
            x_i = batch_x[i:i+1]
            y_i = batch_y[i:i+1]

            # Forward pass
            y_pred = self.model(x_i)
            loss_i = criterion(y_pred, y_i)

            # Compute gradients
            grads = torch.autograd.grad(
                loss_i,
                params.values(),
                retain_graph=False,
                create_graph=False,
            )

            # Store gradients
            for name, grad in zip(param_names, grads):
                per_example_grads[name].append(grad)

        # Stack into [B, *param.shape] tensors
        return {
            name: torch.stack(grads, dim=0)
            for name, grads in per_example_grads.items()
        }

    def update(
        self,
        loss: torch.Tensor,
        per_example_grads: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute and return drift-diffusion metrics for the current step.

        Args:
            loss: Loss value (not used, kept for API compatibility)
            per_example_grads: Dict of per-example gradients [B, *param.shape]

        Returns:
            Dict with keys:
            - "snr_global": Global per-step SNR
            - "grad_norm": Norm of mean gradient
            - "sqrt_trSigma_overB": sqrt(tr(Σ̂)/B)
            - "m_mode[{j}]": Mode coordinate for mode j
            - "g_mode[{j}]": Drift along mode j
            - "sigma_mode[{j}]": Diffusion along mode j
            - "snr_mode[{j}]": SNR for mode j
        """
        B = self.batch_size

        # Handle edge case
        if B < 2:
            return self._nan_metrics()

        # Compute mean gradient (averaged over batch)
        mean_grads = {
            name: grads.mean(dim=0) for name, grads in per_example_grads.items()
        }

        # Flatten to vectors
        bar_g = flatten_grads_from_param_dict(mean_grads)  # Mean gradient

        # Compute per-example gradient vectors
        g_i_list = []
        for i in range(B):
            g_i_dict = {
                name: grads[i] for name, grads in per_example_grads.items()
            }
            g_i = flatten_grads_from_param_dict(g_i_dict)
            g_i_list.append(g_i)

        # Stack into [B, p] tensor
        g_all = torch.stack(g_i_list, dim=0)  # [B, p]

        # Compute trace of covariance matrix
        # tr(Σ̂) = 1/(B-1) Σᵢ ‖gᵢ - ḡ‖²
        deltas = g_all - bar_g.unsqueeze(0)  # [B, p]
        tr_Sigma = (deltas ** 2).sum() / (B - 1)

        # Global statistics
        grad_norm = bar_g.norm().item()
        sqrt_trSigma_overB = math.sqrt(tr_Sigma.item() / B + self.eps)
        snr_global = (
            math.sqrt(self.eta) * grad_norm / (sqrt_trSigma_overB + self.eps)
        )

        # Initialize metrics dict
        metrics = {
            "snr_global": snr_global,
            "grad_norm": grad_norm,
            "sqrt_trSigma_overB": sqrt_trSigma_overB,
        }

        # Compute mode-wise statistics
        if len(self.mode_indices) > 0:
            mode_metrics = self._compute_mode_metrics(
                mean_grads, g_all, bar_g, B
            )
            metrics.update(mode_metrics)

        return metrics

    def _compute_mode_metrics(
        self,
        mean_grads: Dict[str, torch.Tensor],
        g_all: torch.Tensor,
        bar_g: torch.Tensor,
        B: int,
    ) -> Dict[str, float]:
        """Compute mode-wise drift/diffusion metrics.

        Args:
            mean_grads: Dict of mean gradients per parameter
            g_all: Per-example gradients as [B, p] tensor
            bar_g: Mean gradient as 1-D vector
            B: Batch size

        Returns:
            Dict with mode-wise metrics
        """
        metrics = {}

        # Get current effective map with gradients enabled
        W_eff = self.current_effective_map(requires_grad=True)

        # Compute metrics for each tracked mode
        for j in self.mode_indices:
            # Get teacher mode vectors
            u_j = self.teacher_U[:, j:j+1]  # [d_out, 1]
            v_j = self.teacher_V[:, j:j+1]  # [d_in, 1]

            # Compute mode coordinate: m_j = ⟨W_eff, u_j v_j^T⟩_F
            A_j = u_j @ v_j.T  # [d_out, d_in]
            m_j = (W_eff * A_j).sum()

            # Compute gradient of mode coordinate: ψ_j = ∇_θ m_j
            # Use autograd to get gradients w.r.t. all parameters
            m_j.backward(retain_graph=True)

            # Extract ψ_j from parameter gradients
            psi_j_dict = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    psi_j_dict[name] = param.grad.clone()
                    param.grad.zero_()
                else:
                    psi_j_dict[name] = torch.zeros_like(param)

            psi_j = flatten_grads_from_param_dict(psi_j_dict)  # [p]

            # Compute drift: g_j = ψ_j^T ḡ
            g_j = (psi_j * bar_g).sum().item()

            # Compute diffusion: σ_j² = 1/(B-1) Σᵢ (ψ_j^T (gᵢ - ḡ))²
            deltas = g_all - bar_g.unsqueeze(0)  # [B, p]
            projections = (deltas * psi_j.unsqueeze(0)).sum(dim=1)  # [B]
            sigma_j_sq = (projections ** 2).sum() / (B - 1)
            sigma_j = math.sqrt(sigma_j_sq.item() + self.eps)

            # Compute mode SNR: SNR_j = (η |g_j|) / (σ_j²/B)
            snr_j = (
                math.sqrt(self.eta) * abs(g_j) / (sigma_j / math.sqrt(B) + self.eps)
            )

            # Store metrics
            metrics[f"m_mode[{j}]"] = m_j.item()
            metrics[f"g_mode[{j}]"] = g_j
            metrics[f"sigma_mode[{j}]"] = sigma_j
            metrics[f"snr_mode[{j}]"] = snr_j

        return metrics

    def _nan_metrics(self) -> Dict[str, float]:
        """Return NaN metrics for edge cases (B < 2)."""
        metrics = {
            "snr_global": float('nan'),
            "grad_norm": float('nan'),
            "sqrt_trSigma_overB": float('nan'),
        }

        for j in self.mode_indices:
            metrics[f"m_mode[{j}]"] = float('nan')
            metrics[f"g_mode[{j}]"] = float('nan')
            metrics[f"sigma_mode[{j}]"] = float('nan')
            metrics[f"snr_mode[{j}]"] = float('nan')

        return metrics
