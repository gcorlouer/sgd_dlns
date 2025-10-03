import pytest
import torch
from torch import nn
from observables.drift_diffusion import DriftDiffusion
from scripts.models import DLN
from scripts.train import ExperimentConfig
from scripts.teacher import Teacher, TeacherDataset


def test_gradient_norm():
    cfg = ExperimentConfig(
        input_dim=3,
        hidden_dims=5,
        output_dim=3,
        num_hidden_layers=2,
        bias=False,
        lr=0.01,
        rank=2,
        max_singular_value=10,
        min_singular_value=1e-12,
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
        rank=cfg.rank,
        max_singular_value=cfg.max_singular_value,
        min_singular_value=cfg.min_singular_value,
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
    
    obs = DriftDiffusion(model, cfg, dataset, eval_batch_size=10)
    
    batch, target = dataset[:10]
    y = model(batch)
    loss_fn = nn.MSELoss()
    loss_value = loss_fn(y, target)
    model.zero_grad()
    loss_value.backward()
    
    grad_norm = obs.gradient_norm()
    
    # Fixed assertions
    assert grad_norm.dim() == 0, f"grad norm should be scalar, got shape {grad_norm.shape}"
    assert grad_norm.item() >= 0, f"grad norm should be non-negative, got {grad_norm.item()}"


def test_eval_drift():
    cfg = ExperimentConfig(
        input_dim=3,
        hidden_dims=5,
        output_dim=3,
        num_hidden_layers=2,
        bias=False,
        lr=0.01,
        rank=2,
        max_singular_value=10,
        min_singular_value=1e-12,
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
        rank=cfg.rank,
        max_singular_value=cfg.max_singular_value,
        min_singular_value=1e-12,
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
    
    obs = DriftDiffusion(model, cfg, dataset, eval_batch_size=cfg.n_samples)
    eval_drift = obs.drift()
    
    # Manual computation
    batch, target = dataset[:]
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
    
    true_drift = cfg.lr * norm2
    
    assert torch.isclose(eval_drift, true_drift, rtol=1e-4), \
        f"Eval drift {eval_drift} != true drift {true_drift}, diff = {eval_drift - true_drift}"