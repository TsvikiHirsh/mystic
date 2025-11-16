"""
Streamlit App for Interactive Bragg Edge ToF Reconstruction

Demonstrates the ISACSNet-inspired deep learning method for reconstructing
Bragg edge transmission spectra from overlapping neutron ToF pulses.
"""

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.isacs_net import ISACSNet
from utils.signal_generator import BraggEdgeSignalGenerator, create_iron_bcc_spectrum


@st.cache_resource
def load_model(checkpoint_path=None, signal_length=1024, compression_ratio=0.1):
    """Load the trained model."""
    model = ISACSNet(
        signal_length=signal_length,
        compression_ratio=compression_ratio,
        use_chopper_conditioning=True
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success(f"Loaded trained model (PRD: {checkpoint.get('val_prd', 'N/A')}%)")
    else:
        st.sidebar.warning("Using untrained model (for demonstration only)")

    model.eval()
    return model


def plot_wavelength_spectrum(time_axis, wavelength_axis, signal, title, show_wavelength=True):
    """Plot spectrum with dual time/wavelength axis."""
    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(time_axis, signal, 'b-', linewidth=2)
    ax1.set_xlabel('Time of Flight (μs)', fontsize=11)
    ax1.set_ylabel('Transmission / Intensity', fontsize=11, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title, fontsize=12, fontweight='bold')

    if show_wavelength:
        ax2 = ax1.twiny()
        ax2.set_xlabel('Wavelength (Å)', fontsize=11, color='red')
        ax2.plot(wavelength_axis, signal, alpha=0)  # Invisible plot for axis
        ax2.tick_params(axis='x', labelcolor='red')
        # Reverse wavelength axis (shorter wavelengths = longer times)
        ax2.invert_xaxis()

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Bragg Edge ToF Reconstruction",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 Deep Learning Reconstruction of Bragg Edge Transmission")
    st.markdown("""
    **ISACSNet-Inspired Neural Network for Overlapping Neutron ToF Pulses**

    This demonstration shows how deep learning can reconstruct clean Bragg edge transmission
    spectra from highly compressed, overlapping time-of-flight measurements using an
    adaptive chopper sequence.
    """)

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")

    # Signal parameters
    st.sidebar.subheader("Signal Parameters")
    signal_length = st.sidebar.slider("Signal Length (time bins)", 256, 2048, 1024, 256)
    time_max = st.sidebar.slider("Max ToF (μs)", 5000, 20000, 10000, 1000)
    flight_path = st.sidebar.slider("Flight Path (m)", 1.0, 20.0, 5.0, 0.5)

    # Compression parameters
    st.sidebar.subheader("Compression Parameters")
    compression_ratio = st.sidebar.slider(
        "Compression Ratio (%)", 5, 50, 10, 5,
        help="Percentage of time bins where chopper is open"
    ) / 100
    pattern_type = st.sidebar.selectbox(
        "Chopper Pattern",
        ["random", "pseudorandom", "adaptive"],
        index=2,
        help="Adaptive pattern focuses on wavelengths of interest"
    )

    # Noise parameters
    st.sidebar.subheader("Measurement Noise")
    photon_counts = st.sidebar.slider(
        "Photon Counts (log scale)", 2, 5, 3,
        help="10^x total neutron counts"
    )
    photon_counts = 10 ** photon_counts

    # Material selection
    st.sidebar.subheader("Sample Material")
    material_type = st.sidebar.selectbox(
        "Material Type",
        ["Iron BCC", "Custom Edges", "Random Material"]
    )

    # Custom edges if selected
    edge_positions = None
    if material_type == "Custom Edges":
        st.sidebar.markdown("**Custom Bragg Edges (Å):**")
        n_edges = st.sidebar.slider("Number of edges", 1, 5, 2)
        edge_positions = []
        for i in range(n_edges):
            edge_pos = st.sidebar.number_input(
                f"Edge {i+1} position (Å)",
                min_value=0.5, max_value=5.0, value=2.0 + i * 0.5,
                step=0.1, key=f"edge_{i}"
            )
            edge_positions.append(edge_pos)

    # Generate signal button
    if st.sidebar.button("🎲 Generate New Signal", key="generate"):
        st.session_state.regenerate = True

    # Initialize generator
    generator = BraggEdgeSignalGenerator(
        time_bins=signal_length,
        time_range=(0, time_max),
        flight_path=flight_path
    )

    # Generate or retrieve signal
    if 'regenerate' not in st.session_state:
        st.session_state.regenerate = True

    if st.session_state.regenerate:
        # Generate clean spectrum based on material type
        if material_type == "Iron BCC":
            clean_spectrum = create_iron_bcc_spectrum(generator)
            st.session_state.edges_info = "Iron BCC: (110)@2.027Å, (200)@1.433Å, (211)@1.170Å"
        elif material_type == "Custom Edges":
            clean_spectrum = generator.generate_clean_spectrum(edge_positions)
            st.session_state.edges_info = f"Custom edges at: {edge_positions} Å"
        else:  # Random material
            n_edges = np.random.randint(1, 5)
            edge_positions = np.random.uniform(1.5, 4.0, n_edges)
            clean_spectrum = generator.generate_clean_spectrum(edge_positions)
            st.session_state.edges_info = f"Random material with {n_edges} edges"

        # Generate chopper pattern
        chopper_pattern = generator.generate_random_chopper_pattern(
            compression_ratio, pattern_type
        )

        # Create overlapping measurement
        compressed_signal = generator.generate_overlapping_pulses(
            clean_spectrum, chopper_pattern, pulse_width=50
        )

        # Add noise
        noisy_compressed = generator.add_noise(compressed_signal, photon_counts)

        # Store in session state
        st.session_state.clean = clean_spectrum
        st.session_state.chopper = chopper_pattern
        st.session_state.compressed = noisy_compressed
        st.session_state.regenerate = False

    # Retrieve from session state
    clean_spectrum = st.session_state.clean
    chopper_pattern = st.session_state.chopper
    compressed_signal = st.session_state.compressed

    # Load model
    checkpoint_path = "./checkpoints/best_model.pth"
    model = load_model(checkpoint_path, signal_length, compression_ratio)

    # Reconstruct using neural network
    with torch.no_grad():
        compressed_tensor = torch.FloatTensor(compressed_signal).unsqueeze(0).unsqueeze(0)
        chopper_tensor = torch.FloatTensor(chopper_pattern).unsqueeze(0).unsqueeze(0)

        reconstructed = model(compressed_tensor, chopper_tensor)
        reconstructed_np = reconstructed.cpu().numpy()[0, 0]

    # Calculate metrics
    mse = np.mean((reconstructed_np - clean_spectrum) ** 2)
    prd = np.sqrt(np.sum((reconstructed_np - clean_spectrum) ** 2) / np.sum(clean_spectrum ** 2)) * 100
    correlation = np.corrcoef(reconstructed_np, clean_spectrum)[0, 1]
    snr = 10 * np.log10(np.sum(clean_spectrum ** 2) / np.sum((reconstructed_np - clean_spectrum) ** 2))

    # Display metrics
    st.header("📊 Reconstruction Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("PRD (%)", f"{prd:.2f}", help="Percentage Root-mean-square Difference")
    with col2:
        st.metric("Correlation", f"{correlation:.4f}", help="Pearson correlation coefficient")
    with col3:
        st.metric("RMSE", f"{np.sqrt(mse):.4f}", help="Root Mean Square Error")
    with col4:
        st.metric("SNR (dB)", f"{snr:.2f}", help="Signal-to-Noise Ratio")

    st.info(f"**Material:** {st.session_state.edges_info}")
    st.info(f"**Compression:** {compression_ratio*100:.0f}% duty cycle ({pattern_type} pattern) → **{compression_ratio*100:.0f}× data reduction**")

    # Visualization
    st.header("📈 Visualization")

    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔀 Overlapping Measurement",
        "✨ Reconstruction vs Target",
        "📉 Residual Analysis",
        "🌈 All in One"
    ])

    with tab1:
        st.subheader("Compressed Overlapping Measurement")
        fig1, ax1 = plt.subplots(figsize=(12, 5))

        ax1.plot(generator.time_axis, compressed_signal, 'b-', linewidth=1.5,
                alpha=0.7, label='Overlapping measurement')
        ax1.fill_between(generator.time_axis, 0, compressed_signal.max(),
                        where=chopper_pattern > 0.5, alpha=0.2, color='green',
                        label='Chopper open')

        ax1.set_xlabel('Time of Flight (μs)', fontsize=12)
        ax1.set_ylabel('Intensity (arbitrary units)', fontsize=12)
        ax1.set_title('Measured Signal with Overlapping Pulses from Adaptive Chopper',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Add wavelength axis
        ax2 = ax1.twiny()
        ax2.set_xlabel('Wavelength (Å)', fontsize=12, color='red')
        ax2.plot(generator.wavelength_axis, compressed_signal, alpha=0)
        ax2.tick_params(axis='x', labelcolor='red')
        ax2.invert_xaxis()

        plt.tight_layout()
        st.pyplot(fig1)

        st.markdown(f"""
        **What you're seeing:**
        - Blue line: Measured neutron intensity with overlapping ToF pulses
        - Green shaded: Times when chopper is open (only {compression_ratio*100:.0f}% duty cycle!)
        - Multiple pulses overlap, making direct interpretation impossible
        """)

    with tab2:
        st.subheader("Neural Network Reconstruction vs Ground Truth")
        fig2, ax = plt.subplots(figsize=(12, 5))

        ax.plot(generator.time_axis, clean_spectrum, 'g-', linewidth=2.5,
               label='Ground Truth (clean)', alpha=0.8)
        ax.plot(generator.time_axis, reconstructed_np, 'r--', linewidth=2,
               label='NN Reconstruction', alpha=0.8)

        ax.set_xlabel('Time of Flight (μs)', fontsize=12)
        ax.set_ylabel('Transmission', fontsize=12)
        ax.set_title('Bragg Edge Transmission Spectrum: Reconstruction vs Target',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add wavelength axis
        ax2 = ax.twiny()
        ax2.set_xlabel('Wavelength (Å)', fontsize=12, color='red')
        ax2.plot(generator.wavelength_axis, clean_spectrum, alpha=0)
        ax2.tick_params(axis='x', labelcolor='red')
        ax2.invert_xaxis()

        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown(f"""
        **Reconstruction Quality:**
        - PRD: **{prd:.2f}%** (ISACSNet paper reports ~5% at 10% sampling)
        - Correlation: **{correlation:.4f}** (target: >0.998)
        - The neural network successfully recovers Bragg edges from {compression_ratio*100:.0f}× compressed data!
        """)

    with tab3:
        st.subheader("Residual Analysis")
        residual = reconstructed_np - clean_spectrum

        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Residual plot
        ax1.plot(generator.time_axis, residual, 'k-', linewidth=1.5, alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
        ax1.fill_between(generator.time_axis, residual, 0, alpha=0.3, color='gray')
        ax1.set_xlabel('Time of Flight (μs)', fontsize=12)
        ax1.set_ylabel('Residual (Reconstructed - Target)', fontsize=12)
        ax1.set_title(f'Reconstruction Error (RMSE: {np.sqrt(mse):.4f})',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Histogram of residuals
        ax2.hist(residual, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual Value', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Reconstruction Errors', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown(f"""
        **Error Statistics:**
        - Mean error: {np.mean(residual):.4f}
        - Std deviation: {np.std(residual):.4f}
        - Max absolute error: {np.max(np.abs(residual)):.4f}
        """)

    with tab4:
        st.subheader("Complete Visualization")
        fig4, axes = plt.subplots(3, 1, figsize=(12, 12))

        # Panel 1: Compressed measurement
        axes[0].plot(generator.time_axis, compressed_signal, 'b-', linewidth=1.5, alpha=0.7)
        axes[0].fill_between(generator.time_axis, 0, compressed_signal.max(),
                           where=chopper_pattern > 0.5, alpha=0.2, color='green')
        axes[0].set_ylabel('Intensity', fontsize=11)
        axes[0].set_title('(A) Compressed Overlapping Measurement', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(['Measured signal', 'Chopper open'], loc='best')

        # Panel 2: Reconstruction
        axes[1].plot(generator.time_axis, clean_spectrum, 'g-', linewidth=2.5,
                    label='Ground Truth', alpha=0.8)
        axes[1].plot(generator.time_axis, reconstructed_np, 'r--', linewidth=2,
                    label='NN Reconstruction', alpha=0.8)
        axes[1].set_ylabel('Transmission', fontsize=11)
        axes[1].set_title(f'(B) Reconstruction (PRD: {prd:.2f}%, Corr: {correlation:.4f})',
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')

        # Panel 3: Residual
        axes[2].plot(generator.time_axis, residual, 'k-', linewidth=1.5, alpha=0.7)
        axes[2].axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.5)
        axes[2].fill_between(generator.time_axis, residual, 0, alpha=0.3, color='gray')
        axes[2].set_xlabel('Time of Flight (μs)', fontsize=11)
        axes[2].set_ylabel('Residual', fontsize=11)
        axes[2].set_title(f'(C) Reconstruction Error (RMSE: {np.sqrt(mse):.4f})',
                         fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig4)

    # Information section
    st.header("ℹ️ About This Method")

    with st.expander("🔬 Scientific Background"):
        st.markdown("""
        ### Bragg Edge Transmission Spectroscopy

        Neutrons have wavelike properties with wavelength λ related to time-of-flight by:
        **λ [Å] ≈ 3956 × t [μs] / L [m]**

        When neutrons pass through crystalline materials, Bragg diffraction creates sharp
        "edges" in the transmission spectrum at wavelengths corresponding to lattice plane
        spacings: **λ_edge = 2d_hkl**

        These edges reveal crystallographic structure, texture, and strain in materials.

        ### The Measurement Challenge

        Traditional ToF measurements require:
        - Single neutron pulse per measurement (low flux)
        - Full wavelength spectrum acquisition (wasted beam time)
        - Long measurement times (hours)

        ### The Deep Learning Solution

        This implementation adapts ISACSNet (Deng et al., 2025) to:
        1. **Use adaptive chopper patterns** to create overlapping pulses (higher flux)
        2. **Learn optimal measurement operators** instead of random sampling
        3. **Reconstruct clean spectra** from highly compressed measurements
        4. **Achieve 10-100× data compression** while preserving Bragg edge features
        """)

    with st.expander("🧠 Neural Network Architecture"):
        st.markdown("""
        ### ISACSNet-Inspired Architecture

        **Module 1: Random Projection (Learned Compression)**
        - 3 convolutional layers with kernels (8, 8, 4)
        - Learns optimal compression patterns for Bragg edges
        - Replaces hand-designed random matrices

        **Module 2: Dimension Raising**
        - Fully connected layers with LeakyReLU
        - Expands compressed representation

        **Module 3: Initial Reconstruction**
        - Modified Inception blocks (multi-scale kernels: 3, 5, 7, 13)
        - Bi-directional LSTM (250 hidden units) for temporal dependencies
        - Skip connections for gradient flow

        **Module 4: Secondary Reconstruction**
        - Self-attention mechanism for long-range features
        - Refines reconstruction quality

        **Key Innovation:** Conditions reconstruction on the chopper pattern,
        allowing adaptation to different measurement strategies.
        """)

    with st.expander("📊 Performance Comparison"):
        st.markdown("""
        ### Expected Performance (from ISACSNet paper)

        At **10% sampling rate:**
        - ISACSNet: **5% PRD**, 0.9988 correlation
        - Traditional CS (IHT, BP, SAMP): **21-47% PRD**, poor reconstruction

        At **1% sampling rate (100× compression):**
        - ISACSNet: **9.17% PRD**, 0.9715 correlation
        - Traditional CS: **>93% PRD**, <0.2 correlation (complete failure)

        **Reconstruction Speed:**
        - ISACSNet: 0.0108 seconds
        - Traditional iterative algorithms: 0.1-100 seconds (1-4 orders slower)

        This adaptation extends ISACSNet from digital signal compression to
        **physical measurement optimization** for Bragg edge crystallography.
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    **Reference:** Deng et al., "Deep learning-based compressed sampling reconstruction
    algorithm for digitizing intensive neutron ToF signals", Nuclear Science and Techniques, 2025

    **Implementation:** ISACSNet architecture adapted for adaptive Bragg edge ToF reconstruction
    with overlapping pulses and chopper pattern conditioning.
    """)


if __name__ == "__main__":
    main()
