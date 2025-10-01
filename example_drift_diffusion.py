"""Example usage of DriftDiffusionTracker with training loop.

This script demonstrates how to integrate drift-diffusion tracking into
a DLN training loop.
"""

import torch
from torch.utils.data import DataLoader
from scripts.models import DLN
from scripts.teacher import Teacher, TeacherDataset
from metrics.drift_diffusion import DriftDiffusionTracker


def main():
    # Setup
    torch.manual_seed(42)
    device = torch.device('cpu')

    # Model parameters
    d_in, d_out = 10, 10
    widths = [50, 50]
    gamma = 2.5

    # Training parameters
    eta = 1e-4
    batch_size = 4
    num_epochs = 5

    # Create teacher
    teacher = Teacher(
        output_dim=d_out,
        input_dim=d_in,
        rank=4,
        max_singular_value=100.0,
        decay_rate=10.0,
        progression='linear',
        seed=42
    )

    # Get teacher components
    U, S, V = teacher.components
    print(f"Teacher singular values: {S}")

    # Create dataset
    dataset = TeacherDataset(
        teacher,
        n_samples=20,
        noise_std=1.0,
        whiten_inputs=True,
        seed=42
    )

    train_data, test_data = dataset.train_test_split(train_ratio=0.8)

    # Create model
    model = DLN(
        input_dim=d_in,
        hidden_dims=widths,
        output_dim=d_out,
        gamma=gamma,
    ).to(device)

    print(f"\nModel: {len(model.layers)} layers")
    print(f"Total parameters: {model.total_parameters}")

    # Create tracker
    tracker = DriftDiffusionTracker(
        model=model,
        teacher_U=U,
        teacher_V=V,
        eta=eta,
        batch_size=batch_size,
        device=device,
        mode_indices=[0, 1, 2, 3]  # Track top 4 modes
    )

    # Training loop
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)

    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 80)

    for epoch in range(num_epochs):
        # Create data loader
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Compute per-example gradients (BEFORE optimizer step)
            per_ex_grads = tracker.compute_per_example_grads_from_batch(
                x, y, criterion
            )

            # Forward pass
            y_pred = model(x)
            loss = criterion(y_pred, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update tracker and log metrics
            metrics = tracker.update(loss.detach(), per_ex_grads)

            # Print metrics for first batch of each epoch
            if batch_idx == 0:
                print(f"\nEpoch {epoch+1}, Batch {batch_idx+1}:")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  SNR (global): {metrics['snr_global']:.4f}")
                print(f"  Grad norm: {metrics['grad_norm']:.6f}")
                print(f"  sqrt(tr(Σ)/B): {metrics['sqrt_trSigma_overB']:.6f}")

                # Mode-wise metrics
                for j in range(4):
                    print(f"  Mode {j}:")
                    print(f"    m[{j}] = {metrics[f'm_mode[{j}]']:.6f}")
                    print(f"    g[{j}] = {metrics[f'g_mode[{j}]']:.6f}")
                    print(f"    σ[{j}] = {metrics[f'sigma_mode[{j}]']:.6f}")
                    print(f"    SNR[{j}] = {metrics[f'snr_mode[{j}]']:.4f}")

    print("\n" + "-" * 80)
    print("Training complete!")

    # Final effective map
    W_eff = model.effective_map()
    print(f"\nFinal effective map shape: {W_eff.shape}")
    print(f"Final effective rank: {model.effective_rank:.2f}")


if __name__ == "__main__":
    main()
