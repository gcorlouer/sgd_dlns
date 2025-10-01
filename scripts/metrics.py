"""
Observables and metrics for Deep Linear Network experiments.

This module provides utilities for computing various observables during training,
such as mode alignment with teacher singular vectors, effective rank, and more.
"""

import torch
from torch import nn
from scripts.teacher import Teacher


class Observable:
    """Compute observables for DLN training dynamics."""

    def __init__(self, teacher_matrix: Teacher, model: nn.Module):
        self.teacher_matrix = teacher_matrix
        self.model = model

    def weight_matrices_in_io_order(self):
        """Yield Linear weights in input→output order."""
        return [m.weight for m in self.model.modules() if isinstance(m, nn.Linear)]

    def weight_product(self) -> torch.Tensor:
        """
        Returns W_L @ ... @ W_1 (shape: [out_dim, in_dim]) for a stack of Linear layers.
        Uses linalg.multi_dot for speed/associativity.
        """
        with torch.no_grad():
            Ws = self.weight_matrices_in_io_order()
            # Ws is [W1, W2, ..., WL] with shapes [(d1,d0), (d2,d1), ..., (dL,d_{L-1})]
            # Compose into a single matrix mapping input->output: W_L @ ... @ W_1
            if len(Ws) == 0:
                raise ValueError("Model has no nn.Linear layers.")
            if len(Ws) == 1:
                return Ws[0]
            return torch.linalg.multi_dot(Ws[::-1])

    def mode_matrix(self):
        """
        Compute mode alignment matrix between learned network and teacher.

        Projects the learned linear map P onto teacher singular vectors U, V
        to get an (r x r) matrix showing how modes align.

        Returns:
            torch.Tensor: Mode matrix of shape [rank, rank]
        """
        # Assumes teacher.components returns (U, S, V) with shapes
        # U: [out_dim, r], S: [r], V: [in_dim, r]
        U, _, V = self.teacher_matrix.components
        P = self.weight_product()  # [out_dim, in_dim]
        # Project learned map onto teacher singular vectors to get mode-wise matrix
        # result shape: [r, r]
        return U.T @ P @ V
