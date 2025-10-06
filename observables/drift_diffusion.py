"""
Drift-diffusion analysis for stochastic gradient descent in deep linear networks.

This module computes the drift (deterministic gradient) and diffusion (gradient noise)
components of SGD dynamics
during training and compare their relative importance.
"""
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Optional
from scripts.models import DLN
from scripts.config import ExperimentConfig
from scripts.teacher import TeacherDataset


class DriftDiffusion(nn.Module):
    def __init__(self, model: DLN, cfg: ExperimentConfig,
                 dataset: TeacherDataset,
                 loss: Optional[nn.Module] = None,
                 eval_batch_size: int = 1024):
        """
        Initialize drift-diffusion analyzer.

        Args:
            model: Deep linear network to analyze
            cfg: Experiment configuration (contains lr, batch_size)
            dataset: Training dataset
            loss: Loss function (default: MSELoss)
            eval_batch_size: Batch size for empirical gradient estimation (larger = more accurate)
        """
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss = loss if loss is not None else nn.MSELoss()
        self.dataset = dataset
        self.eval_batch_size = eval_batch_size
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device('cpu')

    def gradient_vector(self) -> torch.Tensor:
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                raise ValueError(
                    "Gradients not computed. Call loss.backward() first."
                )
            grads.append(param.grad.flatten())
        return torch.cat(grads)

    def approximate_empirical_gradient(self) -> None:
        """Compute gradient on large random batch."""
        was_training = self.model.training
        self.model.eval()

        self.model.zero_grad()

        n_samples = min(self.eval_batch_size, len(self.dataset))
        indices = torch.randperm(len(self.dataset))[:n_samples].tolist()

        batch, target = self.dataset[indices]
        batch, target = batch.to(self.device), target.to(self.device)
        y = self.model(batch)
        loss = self.loss(y, target)
        loss.backward()

        if was_training:
            self.model.train()

    def gradient_noise_vector(self, empirical_gradient: torch.Tensor,
                              batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient noise: ∇L - ∇L_batch"""
        self.model.zero_grad()
        y = self.model(batch)
        loss = self.loss(y, target)
        loss.backward()

        batch_gradient = self.gradient_vector()

        return empirical_gradient - batch_gradient

    def compute_drift_and_diffusion(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute drift and diffusion in a single pass (efficient).

        Returns:
            drift: η ||∇L||_2 (learning rate × gradient norm)
            diffusion: (η/B) Σ ||∇L - ∇L_batch||_2 (noise magnitude)
        """
        was_training = self.model.training
        self.model.eval()

        # Compute empirical gradient once
        self.approximate_empirical_gradient()
        empirical_gradient = self.gradient_vector()

        # Compute drift from empirical gradient
        drift = self.cfg.lr * torch.linalg.vector_norm(empirical_gradient)

        # Compute diffusion by iterating over batches
        diffusion = torch.tensor(0.0, device=self.device)
        data = DataLoader(
            self.dataset, batch_size=self.cfg.batch_size, shuffle=True
        )
        for batch, target in data:
            batch, target = batch.to(self.device), target.to(self.device)
            noise = self.gradient_noise_vector(empirical_gradient, batch, target)
            diffusion += torch.linalg.vector_norm(noise)
        diffusion = self.cfg.lr / self.cfg.batch_size * diffusion

        if was_training:
            self.model.train()

        return drift, diffusion

    def compute_drift(self) -> torch.Tensor:
        """Compute drift: η ||∇L||_2"""
        drift, _ = self.compute_drift_and_diffusion()
        return drift

    def compute_diffusion(self) -> torch.Tensor:
        """Compute diffusion: (η/B) Σ ||∇L - ∇L_batch||_2"""
        _, diffusion = self.compute_drift_and_diffusion()
        return diffusion

    def drift_diffusion_ratio(self) -> torch.Tensor:
        """Compute drift/diffusion ratio: (η ||∇L||) / ((η/B) Σ ||noise||)"""
        drift, diffusion = self.compute_drift_and_diffusion()
        return drift / (diffusion + 1e-10)  # Add epsilon for numerical stability
