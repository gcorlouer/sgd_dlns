import torch
import torch.nn as nn
from typing import Optional
from scripts.models import DLN
from scripts.train import ExperimentConfig
from scripts.teacher import TeacherDataset


class DriftDiffusion(nn.Module):
    def __init__(self, model: DLN, cfg: ExperimentConfig, dataset: TeacherDataset,
                 loss: Optional[nn.MSELoss] = None, eval_batch_size=1024):
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

    def gradient_norm(self):
        """Compute ||∇L||_2 from current .grad attributes."""
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                raise ValueError(
                    "Gradients not computed. Call loss.backward() first."
                )
            grads.append(param.grad.flatten())

        grad = torch.cat(grads)
        return torch.linalg.vector_norm(grad)

    def approximate_empirical_gradient(self):
        """Compute gradient on large random batch."""
        self.model.zero_grad()

        n_samples = min(self.eval_batch_size, len(self.dataset))
        indices = torch.randperm(len(self.dataset))[:n_samples].tolist()

        batches = []
        targets = []
        for idx in indices:
            b, t = self.dataset[idx]
            batches.append(b)
            targets.append(t)
        batch = torch.stack(batches)
        target = torch.stack(targets)

        batch = batch.to(self.device)
        target = target.to(self.device)

        y = self.model(batch)
        loss = self.loss(y, target)
        loss.backward()

    def drift(self):
        """Compute drift: η ||∇L||_2"""
        self.approximate_empirical_gradient()
        grad_norm = self.gradient_norm()
        return self.cfg.lr * grad_norm

    def diffusion_norm(self):
        pass

    def ratio(self):
        pass
