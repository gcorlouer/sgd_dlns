import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Optional
from scripts.models import DLN
from scripts.train import ExperimentConfig
from scripts.teacher import TeacherDataset


class DriftDiffusion(nn.Module):
    def __init__(self, model: DLN, cfg: ExperimentConfig,
                 dataset:  TeacherDataset,
                 loss: Optional[nn.MSELoss] = None,
                 eval_batch_size=1024):
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

    def gradient_norm(self) -> torch.Tensor:
        """Compute ||∇L||_2 from current .grad attributes."""
        gradient = self.gradient_vector()
        return torch.linalg.vector_norm(gradient)

    def approximate_empirical_gradient(self) -> None:
        """Compute gradient on large random batch."""
        self.model.zero_grad()

        n_samples = min(self.eval_batch_size, len(self.dataset))
        indices = torch.randperm(len(self.dataset))[:n_samples].tolist()

        batch, target = self.dataset[indices]
        batch, target = batch.to(self.device), target.to(self.device)
        y = self.model(batch)
        loss = self.loss(y, target)
        loss.backward()

    def compute_drift(self) -> torch.Tensor:
        """Compute drift: η ||∇L||_2"""
        self.approximate_empirical_gradient()
        grad_norm = self.gradient_norm()
        return self.cfg.lr * grad_norm

    def gradient_noise_vector(self, empirical_gradient: torch.Tensor,
                              batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient noise: ∇L - ∇L_batch"""
        self.model.zero_grad()
        y = self.model(batch)
        loss = self.loss(y, target)
        loss.backward()

        batch_gradient = self.gradient_vector()

        return empirical_gradient - batch_gradient
        
    def compute_diffusion(self) -> torch.Tensor:
        """Compute diffusion: (η/B) Σ ||∇L - ∇L_batch||_2"""
        diffusion = torch.tensor(0.0, device=self.device)
        data = DataLoader(
                self.dataset, batch_size=self.cfg.batch_size, shuffle=True
            )
        self.approximate_empirical_gradient()
        empirical_gradient = self.gradient_vector()
        for batch, target in data:
            batch, target = batch.to(self.device), target.to(self.device)
            noise = self.gradient_noise_vector(empirical_gradient, batch, target)
            diffusion += torch.linalg.vector_norm(noise)
        diffusion = self.cfg.lr / self.cfg.batch_size * diffusion
        return diffusion

    def drift_diffusion_ratio(self) -> torch.Tensor:
        """Compute drift/diffusion ratio: (η ||∇L||) / ((η/B) Σ ||noise||)"""
        drift = self.compute_drift()
        diffusion = self.compute_diffusion()
        return drift / diffusion
