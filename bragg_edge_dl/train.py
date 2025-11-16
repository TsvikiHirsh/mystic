"""
Training script for ISACSNet Bragg Edge Reconstruction

Trains the deep learning network on synthetic overlapping ToF data.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.isacs_net import ISACSNet
from utils.signal_generator import BraggEdgeSignalGenerator, create_dataset


class BraggEdgeDataset(Dataset):
    """PyTorch dataset for Bragg edge ToF signals."""

    def __init__(
        self,
        compressed_signals: np.ndarray,
        target_spectra: np.ndarray,
        chopper_patterns: np.ndarray
    ):
        self.compressed = torch.FloatTensor(compressed_signals).unsqueeze(1)  # Add channel dim
        self.targets = torch.FloatTensor(target_spectra).unsqueeze(1)
        self.choppers = torch.FloatTensor(chopper_patterns).unsqueeze(1)

    def __len__(self):
        return len(self.compressed)

    def __getitem__(self, idx):
        return {
            'compressed': self.compressed[idx],
            'target': self.targets[idx],
            'chopper': self.choppers[idx]
        }


def combined_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    edge_weight: float = 0.5
) -> torch.Tensor:
    """
    Combined loss function for Bragg edge reconstruction.

    Args:
        output: Predicted spectrum
        target: Ground truth spectrum
        mse_weight: Weight for MSE loss
        edge_weight: Weight for edge-preserving loss

    Returns:
        Combined loss value
    """
    # MSE loss for overall reconstruction
    mse_loss = nn.functional.mse_loss(output, target)

    # Edge-preserving loss (gradient matching)
    # This helps preserve sharp Bragg edges
    output_grad = output[:, :, 1:] - output[:, :, :-1]
    target_grad = target[:, :, 1:] - target[:, :, :-1]
    edge_loss = nn.functional.mse_loss(output_grad, target_grad)

    return mse_weight * mse_loss + edge_weight * edge_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_chopper: bool = True
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        compressed = batch['compressed'].to(device)
        target = batch['target'].to(device)
        chopper = batch['chopper'].to(device) if use_chopper else None

        # Forward pass
        output = model(compressed, chopper)

        # Compute loss
        loss = combined_loss(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_chopper: bool = True
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_prd = 0.0  # Percentage root-mean-square difference

    with torch.no_grad():
        for batch in dataloader:
            compressed = batch['compressed'].to(device)
            target = batch['target'].to(device)
            chopper = batch['chopper'].to(device) if use_chopper else None

            # Forward pass
            output = model(compressed, chopper)

            # Compute loss
            loss = combined_loss(output, target)
            total_loss += loss.item()

            # Compute PRD (as used in the paper)
            diff = (output - target) ** 2
            target_power = target ** 2
            prd = torch.sqrt(diff.sum() / target_power.sum()) * 100
            total_prd += prd.item()

    avg_loss = total_loss / len(dataloader)
    avg_prd = total_prd / len(dataloader)

    return avg_loss, avg_prd


def plot_reconstruction_examples(
    model: nn.Module,
    dataset: BraggEdgeDataset,
    device: torch.device,
    n_examples: int = 4,
    save_path: Optional[str] = None
):
    """Plot example reconstructions."""
    model.eval()

    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i in range(n_examples):
            sample = dataset[i]
            compressed = sample['compressed'].unsqueeze(0).to(device)
            target = sample['target'].unsqueeze(0).to(device)
            chopper = sample['chopper'].unsqueeze(0).to(device)

            output = model(compressed, chopper)

            # Convert to numpy
            compressed_np = compressed.cpu().numpy()[0, 0]
            target_np = target.cpu().numpy()[0, 0]
            output_np = output.cpu().numpy()[0, 0]
            chopper_np = chopper.cpu().numpy()[0, 0]

            # Plot compressed signal
            axes[i, 0].plot(compressed_np, 'b-', alpha=0.7, label='Compressed')
            axes[i, 0].fill_between(range(len(chopper_np)), 0, chopper_np.max(),
                                   where=chopper_np > 0.5, alpha=0.2, color='green',
                                   label='Chopper open')
            axes[i, 0].set_title(f'Example {i+1}: Overlapping Measurement')
            axes[i, 0].set_ylabel('Intensity')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)

            # Plot reconstruction vs target
            axes[i, 1].plot(target_np, 'g-', linewidth=2, label='Target (clean)', alpha=0.7)
            axes[i, 1].plot(output_np, 'r--', linewidth=2, label='Reconstructed', alpha=0.7)
            axes[i, 1].set_title('Reconstruction vs Target')
            axes[i, 1].set_ylabel('Transmission')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)

            # Plot residual
            residual = output_np - target_np
            axes[i, 2].plot(residual, 'k-', alpha=0.7)
            axes[i, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[i, 2].set_title(f'Residual (RMSE: {np.sqrt(np.mean(residual**2)):.4f})')
            axes[i, 2].set_ylabel('Difference')
            axes[i, 2].grid(True, alpha=0.3)

            if i == n_examples - 1:
                axes[i, 0].set_xlabel('Time bin')
                axes[i, 1].set_xlabel('Time bin')
                axes[i, 2].set_xlabel('Time bin')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstruction examples to {save_path}")
    else:
        plt.show()

    plt.close()


def train_model(
    signal_length: int = 1024,
    compression_ratio: float = 0.1,
    n_train: int = 5000,
    n_val: int = 500,
    batch_size: int = 16,
    epochs: int = 100,
    learning_rate: float = 0.0005,
    device: str = 'auto',
    save_dir: str = './checkpoints'
):
    """
    Main training function.

    Args:
        signal_length: Length of ToF signal
        compression_ratio: Compression ratio (0.1 = 10% sampling)
        n_train: Number of training samples
        n_val: Number of validation samples
        batch_size: Batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to use ('auto', 'cpu', or 'cuda')
        save_dir: Directory to save checkpoints
    """
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Generate synthetic data
    print("Generating synthetic training data...")
    generator = BraggEdgeSignalGenerator(
        time_bins=signal_length,
        time_range=(0, 10000),
        flight_path=5.0
    )

    duty_cycle = compression_ratio
    train_compressed, train_targets, train_choppers = create_dataset(
        n_train, generator, duty_cycle=duty_cycle, pattern_type='adaptive'
    )

    print("Generating validation data...")
    val_compressed, val_targets, val_choppers = create_dataset(
        n_val, generator, duty_cycle=duty_cycle, pattern_type='adaptive'
    )

    # Create datasets
    train_dataset = BraggEdgeDataset(train_compressed, train_targets, train_choppers)
    val_dataset = BraggEdgeDataset(val_compressed, val_targets, val_choppers)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    print("Creating model...")
    model = ISACSNet(
        signal_length=signal_length,
        compression_ratio=compression_ratio,
        use_chopper_conditioning=True
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_prds = []

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_prd = validate(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_prds.append(val_prd)

        # Update learning rate
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val PRD:    {val_prd:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_prd': val_prd,
            }, checkpoint_path)
            print(f"  Saved best model (PRD: {val_prd:.2f}%)")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss', alpha=0.7)
    ax1.plot(val_losses, label='Val Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_prds, 'b-', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PRD (%)')
    ax2.set_title('Validation PRD over Training')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    print(f"Saved training curves to {save_dir}/training_curves.png")
    plt.close()

    # Plot reconstruction examples
    plot_reconstruction_examples(
        model, val_dataset, device, n_examples=4,
        save_path=os.path.join(save_dir, 'reconstruction_examples.png')
    )

    print(f"\nTraining complete! Best validation PRD: {min(val_prds):.2f}%")
    print(f"Models saved to {save_dir}/")

    return model, train_losses, val_losses, val_prds


if __name__ == "__main__":
    # Train the model
    model, train_losses, val_losses, val_prds = train_model(
        signal_length=1024,
        compression_ratio=0.1,
        n_train=5000,
        n_val=500,
        batch_size=16,
        epochs=100,
        learning_rate=0.0005,
        device='auto',
        save_dir='./checkpoints'
    )

    print("\nFinal Results:")
    print(f"  Final Train Loss: {train_losses[-1]:.6f}")
    print(f"  Final Val Loss:   {val_losses[-1]:.6f}")
    print(f"  Final Val PRD:    {val_prds[-1]:.2f}%")
    print(f"  Best Val PRD:     {min(val_prds):.2f}%")
