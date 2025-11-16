"""
Quick demonstration of Bragg Edge ToF reconstruction using ISACSNet.

This script generates a synthetic measurement and performs reconstruction
without requiring a pre-trained model (uses random initialization for demo).
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.isacs_net import ISACSNet, count_parameters
from utils.signal_generator import BraggEdgeSignalGenerator, create_iron_bcc_spectrum


def main():
    print("=" * 70)
    print("Bragg Edge ToF Reconstruction - Quick Demo")
    print("=" * 70)
    print()

    # Configuration
    signal_length = 1024
    compression_ratio = 0.10  # 10% sampling
    flight_path = 5.0  # meters
    photon_counts = 5000

    print(f"Configuration:")
    print(f"  Signal length: {signal_length} time bins")
    print(f"  Compression ratio: {compression_ratio*100:.0f}% ({1/compression_ratio:.0f}× compression)")
    print(f"  Flight path: {flight_path} m")
    print(f"  Photon counts: {photon_counts}")
    print()

    # Create signal generator
    print("Creating signal generator...")
    generator = BraggEdgeSignalGenerator(
        time_bins=signal_length,
        time_range=(0, 10000),  # microseconds
        flight_path=flight_path
    )

    # Generate Iron BCC spectrum
    print("Generating Iron BCC spectrum...")
    print("  Bragg edges: (110)@2.027Å, (200)@1.433Å, (211)@1.170Å")
    clean_spectrum = create_iron_bcc_spectrum(generator)

    # Generate adaptive chopper pattern
    print(f"Generating adaptive chopper pattern ({compression_ratio*100:.0f}% duty cycle)...")
    chopper_pattern = generator.generate_random_chopper_pattern(
        duty_cycle=compression_ratio,
        pattern_type='adaptive'
    )
    n_open = np.sum(chopper_pattern > 0.5)
    print(f"  Chopper open for {n_open}/{signal_length} time bins")

    # Create overlapping measurement
    print("Simulating overlapping pulses...")
    compressed_signal = generator.generate_overlapping_pulses(
        clean_spectrum, chopper_pattern, pulse_width=50
    )

    # Add Poisson noise
    print(f"Adding Poisson noise (photon counts: {photon_counts})...")
    noisy_compressed = generator.add_noise(compressed_signal, photon_counts)

    # Create model
    print()
    print("Creating ISACSNet model...")
    model = ISACSNet(
        signal_length=signal_length,
        compression_ratio=compression_ratio,
        use_chopper_conditioning=True
    )
    n_params = count_parameters(model)
    print(f"  Model parameters: {n_params:,}")
    print()
    print("  NOTE: Using untrained model with random weights for demonstration.")
    print("        For accurate reconstruction, train the model using train.py")
    print()

    # Reconstruct
    print("Performing reconstruction...")
    model.eval()
    with torch.no_grad():
        compressed_tensor = torch.FloatTensor(noisy_compressed).unsqueeze(0).unsqueeze(0)
        chopper_tensor = torch.FloatTensor(chopper_pattern).unsqueeze(0).unsqueeze(0)

        reconstructed = model(compressed_tensor, chopper_tensor)
        reconstructed_np = reconstructed.cpu().numpy()[0, 0]

    # Calculate metrics
    print("Computing reconstruction metrics...")
    mse = np.mean((reconstructed_np - clean_spectrum) ** 2)
    rmse = np.sqrt(mse)
    prd = np.sqrt(np.sum((reconstructed_np - clean_spectrum) ** 2) /
                  np.sum(clean_spectrum ** 2)) * 100
    correlation = np.corrcoef(reconstructed_np, clean_spectrum)[0, 1]

    print()
    print("Reconstruction Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  PRD:  {prd:.2f}%")
    print(f"  Correlation: {correlation:.4f}")
    print()
    print("  Expected performance with trained model:")
    print("    PRD: ~5-10% at 10% compression")
    print("    Correlation: >0.99")
    print()

    # Create visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Panel 1: Compressed measurement
    axes[0].plot(generator.time_axis, noisy_compressed, 'b-', linewidth=1.5, alpha=0.7,
                label='Overlapping measurement')
    axes[0].fill_between(generator.time_axis, 0, noisy_compressed.max(),
                        where=chopper_pattern > 0.5, alpha=0.2, color='green',
                        label='Chopper open')
    axes[0].set_ylabel('Intensity (arb. units)', fontsize=11)
    axes[0].set_title('(A) Compressed Overlapping ToF Measurement',
                     fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Reconstruction vs target
    axes[1].plot(generator.time_axis, clean_spectrum, 'g-', linewidth=2.5,
                label='Ground Truth', alpha=0.8)
    axes[1].plot(generator.time_axis, reconstructed_np, 'r--', linewidth=2,
                label='NN Reconstruction', alpha=0.8)
    axes[1].set_ylabel('Transmission', fontsize=11)
    axes[1].set_title(f'(B) Reconstruction vs Target (PRD: {prd:.2f}%, Corr: {correlation:.4f})',
                     fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Residual
    residual = reconstructed_np - clean_spectrum
    axes[2].plot(generator.time_axis, residual, 'k-', linewidth=1.5, alpha=0.7)
    axes[2].axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
    axes[2].fill_between(generator.time_axis, residual, 0, alpha=0.3, color='gray')
    axes[2].set_xlabel('Time of Flight (μs)', fontsize=11)
    axes[2].set_ylabel('Residual', fontsize=11)
    axes[2].set_title(f'(C) Reconstruction Error (RMSE: {rmse:.4f})',
                     fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = 'demo_reconstruction.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    print()

    # Summary
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Train the model: python train.py")
    print("  2. Run interactive demo: streamlit run app.py")
    print("  3. View README.md for more information")
    print()
    print("Note: This demo used an untrained model. For accurate reconstruction,")
    print("      train the model on synthetic data using train.py first.")
    print()


if __name__ == "__main__":
    main()
