"""
Tests for Teacher matrix generation and TeacherDataset.
"""

import pytest
import torch
import numpy as np
from scripts.teacher import Teacher, TeacherDataset


class TestTeacherShape:
    """Test that teacher matrices have correct shapes."""

    def test_square_matrix(self):
        """Test square teacher matrix."""
        dim = 10
        rank = 5
        teacher = Teacher(output_dim=dim, input_dim=dim, rank=rank)
        M = teacher.get_matrix

        assert M.shape == (dim, dim), f"Expected shape ({dim}, {dim}), got {M.shape}"

    def test_rectangular_tall_matrix(self):
        """Test rectangular teacher matrix (more outputs than inputs)."""
        output_dim = 20
        input_dim = 10
        rank = 5
        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
        M = teacher.get_matrix

        assert M.shape == (
            output_dim,
            input_dim,
        ), f"Expected shape ({output_dim}, {input_dim}), got {M.shape}"

    def test_rectangular_wide_matrix(self):
        """Test rectangular teacher matrix (more inputs than outputs)."""
        output_dim = 10
        input_dim = 20
        rank = 5
        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
        M = teacher.get_matrix

        assert M.shape == (
            output_dim,
            input_dim,
        ), f"Expected shape ({output_dim}, {input_dim}), got {M.shape}"

    def test_rank_one_matrix(self):
        """Test rank-1 matrix."""
        output_dim = 10
        input_dim = 8
        rank = 1
        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
        M = teacher.get_matrix

        assert M.shape == (output_dim, input_dim)
        # Rank-1 matrix should have only 1 non-zero singular value
        _, S, _ = torch.linalg.svd(M)
        # The first singular value should be much larger than the rest
        # (which should be numerically zero due to rank-1 construction)
        assert S[0] > 1e-5, "First singular value should be non-zero"
        if len(S) > 1:
            # Relative magnitude: S[1] should be negligible compared to S[0]
            assert (
                S[1] / S[0] < 1e-5
            ), f"Second singular value should be negligible: S[0]={S[0]}, S[1]={S[1]}"

    def test_full_rank_square(self):
        """Test full-rank square matrix."""
        dim = 8
        rank = dim  # full rank
        teacher = Teacher(output_dim=dim, input_dim=dim, rank=rank)
        M = teacher.get_matrix

        assert M.shape == (dim, dim)
        _, S, _ = torch.linalg.svd(M)
        nonzero_sv = (S > 1e-10).sum().item()
        assert (
            nonzero_sv == rank
        ), f"Expected rank {rank}, got effective rank {nonzero_sv}"


class TestTeacherSingularValues:
    """Test singular value generation and decomposition."""

    def test_custom_singular_values_identity(self):
        """Test that custom singular values [1,1,...,1] give orthogonal matrix."""
        output_dim = 10
        input_dim = 10
        rank = 10
        custom_sv = torch.ones(rank)

        teacher = Teacher(
            output_dim=output_dim,
            input_dim=input_dim,
            rank=rank,
            custom_singular_values=custom_sv,
        )
        M = teacher.get_matrix

        # Perform SVD
        U_check, S_check, Vt_check = torch.linalg.svd(M, full_matrices=False)

        # Check that singular values are all 1
        assert torch.allclose(
            S_check, torch.ones(rank), atol=1e-5
        ), f"Expected singular values all 1, got {S_check}"

        # Check that M @ M^T ≈ I (orthogonality)
        MMt = M @ M.T
        identity = torch.eye(output_dim)
        assert torch.allclose(
            MMt, identity, atol=1e-4
        ), "M @ M^T should be identity when all singular values are 1"

    def test_custom_singular_values_ordering(self):
        """Test that custom singular values are sorted descending."""
        output_dim = 10
        input_dim = 8
        rank = 5
        # Provide unsorted singular values
        custom_sv = torch.tensor([1.0, 5.0, 2.0, 4.0, 3.0])

        teacher = Teacher(
            output_dim=output_dim,
            input_dim=input_dim,
            rank=rank,
            custom_singular_values=custom_sv,
        )

        U, S, V = teacher.components
        expected_sorted = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        assert torch.allclose(
            S, expected_sorted
        ), f"Expected sorted singular values {expected_sorted}, got {S}"

    def test_power_law_singular_values(self):
        """Test power law singular value progression."""
        output_dim = 10
        input_dim = 10
        rank = 5
        max_sv = 100.0
        decay_rate = 2.0

        teacher = Teacher(
            output_dim=output_dim,
            input_dim=input_dim,
            rank=rank,
            max_singular_value=max_sv,
            decay_rate=decay_rate,
            progression="power",
        )

        _, S, _ = teacher.components

        # Check S[i] = max_sv * i^(-decay_rate)
        expected = torch.tensor(
            [max_sv * (i + 1) ** (-decay_rate) for i in range(rank)]
        )
        assert torch.allclose(
            S, expected, atol=1e-5
        ), f"Power law failed: expected {expected}, got {S}"

    def test_linear_decay_singular_values(self):
        """Test linear decay singular value progression."""
        output_dim = 10
        input_dim = 10
        rank = 5
        max_sv = 50.0
        decay_rate = 5.0

        teacher = Teacher(
            output_dim=output_dim,
            input_dim=input_dim,
            rank=rank,
            max_singular_value=max_sv,
            decay_rate=decay_rate,
            progression="linear",
        )

        _, S, _ = teacher.components

        # Check S[i] = max_sv - decay_rate * i
        expected = torch.tensor([max_sv - decay_rate * i for i in range(rank)])
        expected = torch.clamp(expected, min=1e-12)
        assert torch.allclose(
            S, expected, atol=1e-5
        ), f"Linear decay failed: expected {expected}, got {S}"

    def test_svd_reconstruction(self):
        """Test that M = U @ diag(S) @ V^T."""
        output_dim = 12
        input_dim = 8
        rank = 6

        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
        M = teacher.get_matrix
        U, S, V = teacher.components

        # Reconstruct M from components
        M_reconstructed = U @ torch.diag(S) @ V.T

        assert torch.allclose(
            M, M_reconstructed, atol=1e-5
        ), "Matrix reconstruction from SVD failed"


class TestTeacherOrthogonality:
    """Test that singular vectors U and V are orthonormal."""

    def test_u_orthonormal(self):
        """Test that U^T @ U = I."""
        output_dim = 15
        input_dim = 10
        rank = 8

        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
        U, _, _ = teacher.components

        # U is (output_dim x rank), so U^T @ U should be (rank x rank) identity
        UtU = U.T @ U
        identity = torch.eye(rank)

        assert torch.allclose(
            UtU, identity, atol=1e-5
        ), f"U is not orthonormal: U^T @ U =\n{UtU}"

    def test_v_orthonormal(self):
        """Test that V^T @ V = I."""
        output_dim = 10
        input_dim = 12
        rank = 7

        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
        _, _, V = teacher.components

        # V is (input_dim x rank), so V^T @ V should be (rank x rank) identity
        VtV = V.T @ V
        identity = torch.eye(rank)

        assert torch.allclose(
            VtV, identity, atol=1e-5
        ), f"V is not orthonormal: V^T @ V =\n{VtV}"


class TestTeacherEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_rank_equals_min_dimension(self):
        """Test when rank equals min(output_dim, input_dim)."""
        output_dim = 5
        input_dim = 8
        rank = min(output_dim, input_dim)  # rank = 5

        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
        M = teacher.get_matrix

        assert M.shape == (output_dim, input_dim)
        _, S, _ = torch.linalg.svd(M, full_matrices=False)
        nonzero_sv = (S > 1e-10).sum().item()
        assert (
            nonzero_sv == rank
        ), f"Expected rank {rank}, got effective rank {nonzero_sv}"

    def test_invalid_rank_too_large(self):
        """Test that rank > min(output_dim, input_dim) raises error."""
        output_dim = 5
        input_dim = 8
        rank = 10  # Invalid: rank > min(5, 8)

        with pytest.raises(AssertionError):
            Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)

    def test_invalid_rank_zero(self):
        """Test that rank = 0 raises error."""
        output_dim = 5
        input_dim = 8
        rank = 0

        with pytest.raises(AssertionError):
            Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)

    def test_invalid_rank_negative(self):
        """Test that negative rank raises error."""
        output_dim = 5
        input_dim = 8
        rank = -1

        with pytest.raises(AssertionError):
            Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)

    def test_min_singular_value_clamping(self):
        """Test that singular values are clamped to min_singular_value."""
        output_dim = 10
        input_dim = 10
        rank = 5
        max_sv = 10.0
        min_sv = 0.5
        decay_rate = 100.0  # Very large decay to trigger clamping

        teacher = Teacher(
            output_dim=output_dim,
            input_dim=input_dim,
            rank=rank,
            max_singular_value=max_sv,
            min_singular_value=min_sv,
            decay_rate=decay_rate,
            progression="linear",
        )

        _, S, _ = teacher.components

        # All singular values should be >= min_sv
        assert (
            S >= min_sv
        ).all(), f"Some singular values below min_sv: {S}, min={min_sv}"


class TestTeacherReproducibility:
    """Test reproducibility with seeds."""

    def test_seed_reproducibility(self):
        """Test that same seed produces same teacher matrix."""
        output_dim = 10
        input_dim = 8
        rank = 5
        seed = 42

        teacher1 = Teacher(
            output_dim=output_dim, input_dim=input_dim, rank=rank, seed=seed
        )
        teacher2 = Teacher(
            output_dim=output_dim, input_dim=input_dim, rank=rank, seed=seed
        )

        M1 = teacher1.get_matrix
        M2 = teacher2.get_matrix

        assert torch.allclose(M1, M2), "Same seed should produce identical matrices"

    def test_different_seeds_different_matrices(self):
        """Test that different seeds produce different matrices."""
        output_dim = 10
        input_dim = 8
        rank = 5

        teacher1 = Teacher(
            output_dim=output_dim, input_dim=input_dim, rank=rank, seed=42
        )
        teacher2 = Teacher(
            output_dim=output_dim, input_dim=input_dim, rank=rank, seed=99
        )

        M1 = teacher1.get_matrix
        M2 = teacher2.get_matrix

        assert not torch.allclose(
            M1, M2
        ), "Different seeds should produce different matrices"


class TestTeacherDataset:
    """Test TeacherDataset generation and properties."""

    def test_dataset_shape(self):
        """Test that dataset generates correct shapes."""
        output_dim = 10
        input_dim = 8
        rank = 5
        n_samples = 100

        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
        dataset = TeacherDataset(teacher, n_samples=n_samples)

        assert dataset.X.shape == (
            n_samples,
            input_dim,
        ), f"Expected X shape ({n_samples}, {input_dim}), got {dataset.X.shape}"
        assert dataset.Y.shape == (
            n_samples,
            output_dim,
        ), f"Expected Y shape ({n_samples}, {output_dim}), got {dataset.Y.shape}"

    def test_dataset_output_relationship(self):
        """Test that Y = X @ M^T (without noise)."""
        output_dim = 10
        input_dim = 8
        rank = 5
        n_samples = 50

        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
        dataset = TeacherDataset(teacher, n_samples=n_samples, noise_std=0.0)

        M = teacher.get_matrix
        Y_expected = dataset.X @ M.T

        assert torch.allclose(
            dataset.Y, Y_expected, atol=1e-5
        ), "Dataset Y should equal X @ M^T when noise_std=0"

    def test_dataset_train_test_split(self):
        """Test train/test split proportions."""
        output_dim = 10
        input_dim = 8
        rank = 5
        n_samples = 100
        train_ratio = 0.8

        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank)
        dataset = TeacherDataset(teacher, n_samples=n_samples)
        train, test = dataset.train_test_split(train_ratio=train_ratio)

        expected_train = int(n_samples * train_ratio)
        expected_test = n_samples - expected_train

        assert (
            train.n_samples == expected_train
        ), f"Expected {expected_train} train samples, got {train.n_samples}"
        assert (
            test.n_samples == expected_test
        ), f"Expected {expected_test} test samples, got {test.n_samples}"

    def test_dataset_with_noise(self):
        """Test that noise is added when noise_std > 0."""
        output_dim = 10
        input_dim = 8
        rank = 5
        n_samples = 50
        noise_std = 1.0

        teacher = Teacher(output_dim=output_dim, input_dim=input_dim, rank=rank, seed=42)
        dataset = TeacherDataset(
            teacher, n_samples=n_samples, noise_std=noise_std, seed=42
        )

        M = teacher.get_matrix
        Y_clean = dataset.X @ M.T

        # With noise, Y should NOT equal X @ M^T exactly
        assert not torch.allclose(
            dataset.Y, Y_clean
        ), "Y should differ from X @ M^T when noise is added"

        # But they should be close (within a few standard deviations)
        diff = (dataset.Y - Y_clean).abs()
        # Expect most differences to be within 3 * noise_std
        assert (
            diff < 3 * noise_std
        ).float().mean() > 0.9, "Noise appears too large or structured"

    def test_dataset_reproducibility(self):
        """Test that dataset generation is reproducible with seed."""
        output_dim = 10
        input_dim = 8
        rank = 5
        n_samples = 50
        seed = 123

        teacher = Teacher(
            output_dim=output_dim, input_dim=input_dim, rank=rank, seed=seed
        )
        dataset1 = TeacherDataset(teacher, n_samples=n_samples, seed=seed)

        # Reset teacher and dataset with same seed
        teacher2 = Teacher(
            output_dim=output_dim, input_dim=input_dim, rank=rank, seed=seed
        )
        dataset2 = TeacherDataset(teacher2, n_samples=n_samples, seed=seed)

        assert torch.allclose(
            dataset1.X, dataset2.X
        ), "Same seed should produce identical X"
        assert torch.allclose(
            dataset1.Y, dataset2.Y
        ), "Same seed should produce identical Y"
