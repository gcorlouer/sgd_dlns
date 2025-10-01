"""
Plotting utilities for Deep Linear Network experiments.

This module provides functions for visualizing training dynamics,
including loss curves, mode evolution, and alignment metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional


def plot_loss_curves(
    train_losses: List[float],
    test_losses: List[float],
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot training and test loss curves.

    Args:
        train_losses: List of training losses over epochs
        test_losses: List of test losses over epochs
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    iterations = np.arange(len(train_losses))
    fig = plt.figure()
    plt.plot(iterations, train_losses, label="Train loss")
    plt.plot(iterations, test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    return fig


def plot_diagonal_modes(
    modes: List[torch.Tensor],
    rank: int,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot evolution of diagonal mode elements.

    Args:
        modes: List of mode matrices (each [rank x rank])
        rank: Number of modes to plot
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    iterations = np.arange(len(modes))
    diag_modes = torch.stack([torch.diag(m) for m in modes])

    fig = plt.figure()
    for i in range(rank):
        plt.plot(iterations, diag_modes[:, i].numpy(), label=f"mode {i}")
    plt.xlabel("Epoch")
    plt.ylabel("Diagonal mode strength")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    return fig


def plot_off_diagonal_modes(
    modes: List[torch.Tensor], save_path: Optional[Path] = None, show: bool = True
) -> plt.Figure:
    """
    Plot evolution of off-diagonal mode elements (upper triangular mean).

    Args:
        modes: List of mode matrices
        save_path: Optional path to save the figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    iterations = np.arange(len(modes))
    off_diag_modes = np.stack(
        [torch.mean(torch.triu(m, diagonal=1)).numpy() for m in modes]
    )

    fig = plt.figure()
    plt.plot(iterations, off_diag_modes, label="off-diagonal mean")
    plt.xlabel("Epoch")
    plt.ylabel("Off-diagonal mode strength")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

    return fig
