import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class DLN(nn.Module):
    """Deep Linear Network implementation.

    A neural network with multiple linear layers and no nonlinearities.
    Useful for studying SGD dynamics and rank constraints in linear settings.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions, or int for uniform hidden layers
        output_dim: Output dimension  
        num_hidden_layers: Number of hidden layers (ignored if hidden_dims is list)
        gamma: Initialization exponent
        bias: Whether to include bias terms
    """
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: int | list[int],
        output_dim: int,
        num_hidden_layers: int = 3,
        gamma: float = 1.5,
        bias: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma  # γ > 1 for saddle-to-saddle regime
        
        # Handle hidden dimensions specification
        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * num_hidden_layers
        else:
            self.hidden_dims = hidden_dims
            
        # Build layer dimensions
        layer_dims = [input_dim] + self.hidden_dims + [output_dim]
        
        # Create layers with proper initialization
        self.layers = nn.ModuleList([
            self._linear_layer(layer_dims[i], layer_dims[i+1], bias)
            for i in range(len(layer_dims) - 1)
        ])
        
        self._initialize_weights()
    
    def _linear_layer(self, in_features: int, out_features: int, bias: bool) -> nn.Linear:
        """Create a single linear layer."""
        return nn.Linear(in_features, out_features, bias=bias)
    
    def _initialize_weights(self):
        """Initialize weights using σ² = w^(-γ) scaling."""
        # Paper uses rectangular networks where all hidden layers have same width
        width = self.hidden_dims[0]
        
        # Paper's exact formula: σ² = w^(-γ)
        variance = width ** (-self.gamma)
        std = variance ** 0.5
        
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all linear layers."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    @property
    def total_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property 
    def effective_rank(self) -> float:
        """Compute effective rank of the composed linear transformation.
        
        For a DLN W_L @ ... @ W_1, the effective rank is bounded by
        min(rank(W_i)) across all layers.
        """
        with torch.no_grad():
            # Compute composition W_L @ ... @ W_1
            result = self.layers[0].weight
            for layer in self.layers[1:]:
                result = layer.weight @ result
            
            # SVD to get singular values
            _, s, _ = torch.svd(result)
            
            # Effective rank (participation ratio of singular values)
            rank = (s ** 2).sum() / s.max() ** 2
            return rank
    
    def get_layer_ranks(self) -> list[int]:
        """Get rank of each individual layer."""
        with torch.no_grad():
            return [torch.linalg.matrix_rank(layer.weight).item() for layer in self.layers]

if __name__ == '__main__':
    # Test the implementation
    model = DLN(
        input_dim=5,
        hidden_dims=10, 
        output_dim=3,
        num_hidden_layers=4,
        gamma=1.5
    )
    
    # Test forward pass
    x = torch.randn(32, 5)  # batch_size=32
    output = model(x)
    
    print(f"Model architecture: {[10] + model.hidden_dims + [3]}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.total_parameters}")
    print(f"Layer ranks: {model.get_layer_ranks()}")
    print(f"Effective rank: {model.effective_rank:.2f}")