import torch
from torch import Tensor
from typing import Optional, Callable, Literal
from enum import Enum

class Teacher():

    def __init__(self, 
                 output_dim: int, 
                 input_dim: int, 
                 rank: int,
                 seed: Optional[int] = None,
                 max_singular_value: float = 1.0,
                 min_singular_value: float = 1e-12,
                 decay_rate: float = 2.0,
                 progression: str = 'power',
                 custom_singular_values: Optional[Tensor] = None):
        """
        Args:
            output_dim (m): Output dimension
            input_dim (n): Input dimension  
            rank (r): Matrix rank (must be ≤ min(m,n))
            seed: Random seed for reproducibility
        """
        assert 1 <= rank <= min(output_dim, input_dim), f"Rank {rank} must satisfy 1 ≤ r ≤ min({output_dim}, {input_dim})"
            
        self.output_dim = output_dim  # m
        self.input_dim = input_dim    # n  
        self.rank = rank              # r
        self.max_singular_value: float = max_singular_value
        self.min_singular_value: float = min_singular_value
        self.decay_rate: float = decay_rate
        self.custom_singular_values: Optional[Tensor] = custom_singular_values
        self.progression: str = progression
        
        if seed is not None:
            torch.manual_seed(seed)
        self._U = None
        self._V = None
        self._S = None
        self._M = None
        self.matrix()

    def singular_vectors(self) -> tuple[Tensor, Tensor]:
        """
        Generate orthonormal singular vectors U, V via QR decomposition.
        
        Returns:
            U ∈ R^(m×r), V ∈ R^(n×r) with orthonormal columns
        """
        # Standard Gaussian initialization + QR orthogonalization
        U_raw = torch.randn(self.output_dim, self.rank)
        V_raw = torch.randn(self.input_dim, self.rank)
        
        U, _ = torch.linalg.qr(U_raw)
        V, _ = torch.linalg.qr(V_raw)
            
        self._U = U
        self._V = V

    def singular_values(self):
        """
        Generate singular values with specified distribution.
        
        Args:
            max_val: Maximum singular value (σ_1)  
            alpha: Power law exponent or exponential decay rate
            custom_values: User-provided singular values
            
        Returns:
            S ∈ R^r: singular values in descending order
        """
        if self.custom_singular_values is not None:
            S = self.custom_singular_values.clone()
            assert len(S) == self.rank, f"Expected {self.rank} values, got {len(S)}"
            S, _ = torch.sort(S, descending=True)
        elif self.progression == 'power':
            indices = torch.arange(1, self.rank + 1, dtype=torch.float32)
            S = self.max_singular_value * (indices ** (-self.decay_rate))
            S = torch.clamp(S, min=self.min_singular_value)
            self._S = S
        elif self.progression == 'linear':
            indices = torch.arange(0, self.rank, dtype=torch.float32)
            S = self.max_singular_value - self.decay_rate * indices
            S = torch.clamp(S, min=self.min_singular_value)
            self._S = S
        return self._S

    def matrix(self):
        self.singular_vectors()
        self.singular_values()
        # M = U @ diag(S) @ V^T
        self._M = self._U @ torch.diag(self._S) @ self._V.T
        return self._M

    @property 
    def get_matrix(self) -> Tensor:
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
        M = self.get_matrix
        _, S, _ = torch.linalg.svd(M, full_matrices=False)
        return (S >= threshold * S[0]).sum().item()
    
    def spectral_decay_rate(self) -> float:
        """Estimate exponential decay rate: α such that σ_i ≈ σ_1 * exp(-αi)"""
        if self._S is None or self.rank < 2:
            return float('nan')
        log_ratios = torch.log(self._S[:-1] / self._S[1:])
        return log_ratios.mean().item()

from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

class TeacherDataset(Dataset):
    def __init__(self, 
                 teacher: Teacher,
                 n_samples: int = 1000,
                 noise_std: float = 0.0,
                 whiten_inputs: bool = False,
                 seed: Optional[int] = None):
        self.teacher = teacher
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.whiten_inputs = whiten_inputs
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.X, self.Y = self._generate_data()
    
    def _generate_data(self) -> Tuple[Tensor, Tensor]:
        """Generate (X, Y) pairs where Y = MX + noise"""
        # Standard shape: (n_samples, input_dim)
        X = torch.randn(self.n_samples, self.teacher.input_dim)
        
        if self.whiten_inputs:
            X = self._whiten(X)
            
        # Y = X @ M^T
        Y = X @ self.teacher.matrix().T
        
        if self.noise_std > 0:
            Y += torch.randn_like(Y) * self.noise_std
            
        return X, Y
    
    def _whiten(self, X: Tensor) -> Tensor:
        """Whiten input data"""
        # Center, then thin SVD on (n_samples x input_dim)
        Xc = X - X.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        # PCA whitening: preserve shape (n_samples, input_dim)
        X_white = U @ Vh * (self.n_samples ** 0.5)
        return X_white
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.X[idx], self.Y[idx]
    
    def train_test_split(self, train_ratio: float = 0.8) -> Tuple['TeacherDataset', 'TeacherDataset']:
        """Split into train/test datasets"""
        n_train = int(self.n_samples * train_ratio)
        indices = torch.randperm(self.n_samples)
        
        train_dataset = TeacherDataset.__new__(TeacherDataset)
        train_dataset.__dict__ = self.__dict__.copy()
        train_dataset.X = self.X[indices[:n_train]]
        train_dataset.Y = self.Y[indices[:n_train]]
        train_dataset.n_samples = n_train
        
        test_dataset = TeacherDataset.__new__(TeacherDataset)
        test_dataset.__dict__ = self.__dict__.copy()
        test_dataset.X = self.X[indices[n_train:]]
        test_dataset.Y = self.Y[indices[n_train:]]
        test_dataset.n_samples = self.n_samples - n_train
        
        return train_dataset, test_dataset

if __name__ == "__main__":
    output_dim = 10
    input_dim = 10
    rank = 3
    teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
    _, S, _ = teacher.components
    print(f"Teacher matrix is {teacher.get_matrix}")
    print(f"Teacher matrix shape is {teacher.get_matrix.shape}")
    print(f"Singular values are {S}")
    print(f"Rank is {teacher.effective_rank()} should be {rank}")
    # Generate dataset
    dataset = TeacherDataset(teacher, n_samples=10)
    train, test = dataset.train_test_split()
    print(train.X.shape)
    print(train.Y.shape)

