import torch
from torch import Tensor
from typing import Optional, Callable, Literal
from enum import Enum
import numpy as np

class SingularValueDistribution(Enum):
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    UNIFORM = "uniform" 
    CUSTOM = "custom"

class TeacherMatrixGenerator:
    """
    Research-grade teacher matrix generator for studying SGD dynamics.
    
    Constructs M = U @ S @ V^T where:
    - U ∈ R^(m×r): left singular vectors (orthonormal columns)
    - S ∈ R^(r×r): diagonal singular values
    - V ∈ R^(n×r): right singular vectors (orthonormal columns)
    - M ∈ R^(m×n): resulting teacher matrix of rank r
    """
    
    def __init__(self, 
                 output_dim: int, 
                 input_dim: int, 
                 rank: int,
                 seed: Optional[int] = None):
        """
        Args:
            output_dim (m): Output dimension
            input_dim (n): Input dimension  
            rank (r): Matrix rank (must be ≤ min(m,n))
            seed: Random seed for reproducibility
        """
        assert 1 <= rank <= min(output_dim, input_dim), \
            f"Rank {rank} must satisfy 1 ≤ r ≤ min({output_dim}, {input_dim})"
            
        self.output_dim = output_dim  # m
        self.input_dim = input_dim    # n  
        self.rank = rank              # r
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self._U = None
        self._S = None  
        self._V = None
        self._M = None
    
    def generate_singular_vectors(self, 
                                distribution: Literal["gaussian", "uniform"] = "gaussian",
                                orthogonalization: Literal["qr", "svd"] = "qr") -> tuple[Tensor, Tensor]:
        """
        Generate orthonormal singular vectors U, V.
        
        Args:
            distribution: Initial random distribution before orthogonalization
            orthogonalization: Method for ensuring orthonormality
            
        Returns:
            U ∈ R^(m×r), V ∈ R^(n×r) with orthonormal columns
        """
        if distribution == "gaussian":
            U_raw = torch.randn(self.output_dim, self.rank)
            V_raw = torch.randn(self.input_dim, self.rank)
        elif distribution == "uniform":
            U_raw = torch.rand(self.output_dim, self.rank) * 2 - 1
            V_raw = torch.rand(self.input_dim, self.rank) * 2 - 1
            
        if orthogonalization == "qr":
            U, _ = torch.linalg.qr(U_raw)
            V, _ = torch.linalg.qr(V_raw)
        elif orthogonalization == "svd":
            # More numerically stable for ill-conditioned matrices
            U_svd, _, _ = torch.linalg.svd(U_raw, full_matrices=False)
            V_svd, _, _ = torch.linalg.svd(V_raw, full_matrices=False)  
            U = U_svd[:, :self.rank]
            V = V_svd[:, :self.rank]
            
        self._U = U
        self._V = V
        return U, V
    
    def generate_singular_values(self,
                               distribution: SingularValueDistribution = SingularValueDistribution.EXPONENTIAL,
                               max_val: float = 1.0,
                               min_val: float = 0.01,
                               alpha: float = 2.0,
                               custom_values: Optional[Tensor] = None) -> Tensor:
        """
        Generate singular values with specified distribution.
        
        Args:
            distribution: Type of singular value distribution
            max_val: Maximum singular value (σ_1)
            min_val: Minimum singular value (σ_r)  
            alpha: Power law exponent or exponential decay rate
            custom_values: User-provided singular values
            
        Returns:
            S ∈ R^r: singular values in descending order
        """
        if distribution == SingularValueDistribution.CUSTOM:
            assert custom_values is not None, "Must provide custom_values"
            S = custom_values.clone()
            assert len(S) == self.rank, f"Expected {self.rank} values, got {len(S)}"
        
        elif distribution == SingularValueDistribution.EXPONENTIAL:
            # σ_i = σ_1 * exp(-α * (i-1)/(r-1)) 
            if self.rank == 1:
                S = torch.tensor([max_val])
            else:
                decay_factors = torch.linspace(0, alpha, self.rank)
                S = max_val * torch.exp(-decay_factors)
                S = S * (min_val / S[-1])  # Ensure min value
        
        elif distribution == SingularValueDistribution.POWER_LAW:
            # σ_i = σ_1 * (i^(-1/α))
            indices = torch.arange(1, self.rank + 1, dtype=torch.float32)
            S = max_val * (indices ** (-1/alpha))
            S = S * (min_val / S[-1])  # Rescale to ensure min value
            
        elif distribution == SingularValueDistribution.UNIFORM:
            S = torch.linspace(max_val, min_val, self.rank)
        
        # Ensure descending order
        S, _ = torch.sort(S, descending=True)
        self._S = S
        return S
    
    def construct_teacher_matrix(self, 
                               U: Optional[Tensor] = None,
                               S: Optional[Tensor] = None, 
                               V: Optional[Tensor] = None) -> Tensor:
        """
        Construct teacher matrix M = U @ diag(S) @ V^T.
        
        Args:
            U, S, V: Optional override components (uses stored if None)
            
        Returns:
            M ∈ R^(m×n): teacher matrix
        """
        U = U if U is not None else self._U
        S = S if S is not None else self._S  
        V = V if V is not None else self._V
        
        assert all(x is not None for x in [U, S, V]), \
            "Must generate or provide U, S, V before constructing M"
            
        # M = U @ diag(S) @ V^T
        self._M = U @ torch.diag(S) @ V.T
        return self._M
    
    def generate_full_matrix(self,
                           sv_distribution: SingularValueDistribution = SingularValueDistribution.EXPONENTIAL,
                           vec_distribution: Literal["gaussian", "uniform"] = "gaussian",
                           **sv_kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generate complete SVD decomposition: M = U @ diag(S) @ V^T.
        
        Returns:
            M, U, S, V: teacher matrix and its SVD components
        """
        U, V = self.generate_singular_vectors(distribution=vec_distribution)
        S = self.generate_singular_values(distribution=sv_distribution, **sv_kwargs)
        M = self.construct_teacher_matrix(U, S, V)
        
        return M, U, S, V
    
    @property 
    def teacher_matrix(self) -> Tensor:
        """Get the constructed teacher matrix"""
        if self._M is None:
            raise ValueError("Teacher matrix not yet constructed")
        return self._M
    
    @property
    def components(self) -> tuple[Tensor, Tensor, Tensor]:
        """Get (U, S, V) components"""
        return self._U, self._S, self._V
    
    def condition_number(self) -> float:
        """Compute κ(M) = σ_max / σ_min"""
        if self._S is None:
            raise ValueError("Singular values not generated")
        return (self._S[0] / self._S[-1]).item()
    
    def effective_rank(self, threshold: float = 1e-6) -> int:
        """Compute effective rank: |{i : σ_i ≥ threshold * σ_1}|"""
        if self._S is None:
            raise ValueError("Singular values not generated")
        return (self._S >= threshold * self._S[0]).sum().item()
    
    def spectral_decay_rate(self) -> float:
        """Estimate exponential decay rate: α such that σ_i ≈ σ_1 * exp(-αi)"""
        if self._S is None or self.rank < 2:
            return float('nan')
        log_ratios = torch.log(self._S[:-1] / self._S[1:])
        return log_ratios.mean().item()

class DataGenerator:
    """Generate training data from teacher matrix"""
    
    def __init__(self, teacher_generator: TeacherMatrixGenerator):
        self.teacher = teacher_generator
    
    def generate_batch(self, 
                      batch_size: int,
                      input_distribution: Literal["gaussian", "uniform", "sphere"] = "gaussian",
                      noise_std: float = 0.0,
                      input_scale: float = 1.0) -> tuple[Tensor, Tensor]:
        """
        Generate batch of training data (X, Y) where Y = MX + noise.
        
        Args:
            batch_size: Number of samples
            input_distribution: Distribution of input vectors
            noise_std: Standard deviation of additive noise
            input_scale: Scaling factor for inputs
            
        Returns:
            X ∈ R^(n×B), Y ∈ R^(m×B): input and output batches
        """
        if input_distribution == "gaussian":
            X = torch.randn(self.teacher.input_dim, batch_size) * input_scale
        elif input_distribution == "uniform":  
            X = (torch.rand(self.teacher.input_dim, batch_size) * 2 - 1) * input_scale
        elif input_distribution == "sphere":
            # Uniform on unit sphere, then scaled
            X = torch.randn(self.teacher.input_dim, batch_size)
            X = X / torch.norm(X, dim=0, keepdim=True) * input_scale
            
        Y_clean = self.teacher.teacher_matrix @ X
        
        if noise_std > 0:
            noise = torch.randn_like(Y_clean) * noise_std  
            Y = Y_clean + noise
        else:
            Y = Y_clean
            
        return X, Y

# Research utilities
def analyze_teacher_matrix(M: Tensor, U: Tensor, S: Tensor, V: Tensor) -> dict:
    """Comprehensive analysis of teacher matrix properties"""
    return {
        'shape': M.shape,
        'rank': len(S),
        'condition_number': (S[0] / S[-1]).item(),
        'spectral_norm': S[0].item(),
        'nuclear_norm': S.sum().item(),
        'frobenius_norm': torch.norm(M, 'fro').item(),
        'singular_values': S.tolist(),
        'orthogonality_error_U': torch.norm(U.T @ U - torch.eye(U.shape[1])).item(),
        'orthogonality_error_V': torch.norm(V.T @ V - torch.eye(V.shape[1])).item(),
        'reconstruction_error': torch.norm(M - U @ torch.diag(S) @ V.T).item()
    }

# Example usage for SGD research
if __name__ == "__main__":
    # Generate teacher matrix with exponential singular value decay  
    generator = TeacherMatrixGenerator(output_dim=20, input_dim=50, rank=5, seed=42)
    M, U, S, V = generator.generate_full_matrix(
        sv_distribution=SingularValueDistribution.EXPONENTIAL,
        max_val=1.0, 
        min_val=0.01,
        alpha=3.0
    )
    
    # Analysis
    analysis = analyze_teacher_matrix(M, U, S, V)
    print("Teacher Matrix Analysis:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Generate training data
    data_gen = DataGenerator(generator)
    X_train, Y_train = data_gen.generate_batch(
        batch_size=1000,
        input_distribution="gaussian", 
        noise_std=0.01
    )
    
    print(f"\nTraining data: X{X_train.shape}, Y{Y_train.shape}")
    print(f"Data SNR: {torch.norm(M @ X_train)**2 / torch.norm(Y_train - M @ X_train)**2:.1f}")