"""
Synthetic Bragg Edge ToF Signal Generator

This module generates synthetic neutron time-of-flight signals with Bragg edges
for training and testing the deep learning reconstruction network.
"""

import numpy as np
from typing import List, Tuple, Optional


class BraggEdgeSignalGenerator:
    """
    Generate synthetic Bragg edge transmission spectra for neutron ToF measurements.

    Bragg edges appear as step-like decreases in transmission at specific wavelengths
    corresponding to crystallographic planes in the material.
    """

    def __init__(
        self,
        time_bins: int = 1024,
        time_range: Tuple[float, float] = (0, 10000),  # microseconds
        flight_path: float = 5.0,  # meters
    ):
        """
        Initialize the signal generator.

        Args:
            time_bins: Number of time bins in the ToF spectrum
            time_range: (min, max) time in microseconds
            flight_path: Flight path length in meters
        """
        self.time_bins = time_bins
        self.time_range = time_range
        self.flight_path = flight_path

        # Create time axis
        self.time_axis = np.linspace(time_range[0], time_range[1], time_bins)

        # Constants for wavelength conversion
        # λ = h*t / (m_n * L) where h is Planck constant, m_n is neutron mass
        # λ[Å] ≈ 3956 * t[μs] / L[m]
        self.wavelength_axis = 3956 * self.time_axis / self.flight_path

    def _bragg_edge(self, wavelength: np.ndarray, edge_lambda: float,
                    edge_height: float = 0.3, edge_width: float = 0.05) -> np.ndarray:
        """
        Generate a single Bragg edge feature.

        The edge is modeled as a smooth step function using tanh.

        Args:
            wavelength: Wavelength array in Angstroms
            edge_lambda: Edge position in Angstroms
            edge_height: Height of the edge step (0 to 1)
            edge_width: Width parameter controlling edge sharpness

        Returns:
            Transmission attenuation due to this edge
        """
        # Smooth step function centered at edge_lambda
        return edge_height * (1 + np.tanh((wavelength - edge_lambda) / edge_width)) / 2

    def generate_clean_spectrum(
        self,
        edge_positions: List[float],
        edge_heights: Optional[List[float]] = None,
        edge_widths: Optional[List[float]] = None,
        baseline: float = 1.0,
        add_absorption: bool = True
    ) -> np.ndarray:
        """
        Generate a clean Bragg edge transmission spectrum.

        Args:
            edge_positions: List of Bragg edge positions in Angstroms
            edge_heights: List of edge heights (default: 0.2-0.4 random)
            edge_widths: List of edge widths (default: 0.03-0.08 random)
            baseline: Baseline transmission (1.0 = 100%)
            add_absorption: Add smooth absorption background

        Returns:
            Transmission spectrum (time_bins,)
        """
        n_edges = len(edge_positions)

        if edge_heights is None:
            edge_heights = np.random.uniform(0.2, 0.4, n_edges)
        if edge_widths is None:
            edge_widths = np.random.uniform(0.03, 0.08, n_edges)

        # Start with baseline
        transmission = np.ones(self.time_bins) * baseline

        # Add smooth absorption background (1/v absorption)
        if add_absorption:
            # Typical 1/v absorption decreases transmission at longer wavelengths
            absorption = 0.1 * (self.wavelength_axis / self.wavelength_axis.mean()) ** 0.5
            transmission -= absorption

        # Add each Bragg edge
        for edge_pos, edge_h, edge_w in zip(edge_positions, edge_heights, edge_widths):
            edge_feature = self._bragg_edge(self.wavelength_axis, edge_pos, edge_h, edge_w)
            transmission -= edge_feature

        # Ensure transmission is in valid range
        transmission = np.clip(transmission, 0.01, 1.0)

        return transmission

    def add_noise(
        self,
        signal: np.ndarray,
        photon_counts: float = 10000,
        background_rate: float = 10,
    ) -> np.ndarray:
        """
        Add Poisson noise to simulate counting statistics.

        Args:
            signal: Clean transmission signal
            photon_counts: Average total neutron counts
            background_rate: Background counts per bin

        Returns:
            Noisy signal with Poisson statistics
        """
        # Convert transmission to counts
        expected_counts = signal * photon_counts / self.time_bins
        expected_counts += background_rate

        # Apply Poisson noise
        noisy_counts = np.random.poisson(expected_counts)

        # Convert back to transmission (with background subtraction)
        noisy_signal = (noisy_counts - background_rate) * self.time_bins / photon_counts

        return np.clip(noisy_signal, 0.01, 1.5)

    def generate_overlapping_pulses(
        self,
        clean_spectrum: np.ndarray,
        chopper_pattern: np.ndarray,
        pulse_width: int = 50,
    ) -> np.ndarray:
        """
        Simulate overlapping pulses from an adaptive chopper sequence.

        This simulates the FOBI-like measurement where multiple pulses overlap
        due to a coded chopper pattern.

        Args:
            clean_spectrum: Clean transmission spectrum
            chopper_pattern: Binary array indicating when chopper is open (1) or closed (0)
            pulse_width: Width of each neutron pulse in time bins

        Returns:
            Measured signal with overlapping pulses
        """
        # Normalize chopper pattern to binary
        chopper_binary = (chopper_pattern > 0.5).astype(float)

        # Create pulse shape (Gaussian)
        pulse_shape = np.exp(-np.linspace(-3, 3, pulse_width)**2 / 2)
        pulse_shape /= pulse_shape.sum()

        # Convolve spectrum with pulse shape and chopper pattern
        measured = np.zeros_like(clean_spectrum)

        for i in range(self.time_bins):
            if chopper_binary[i] > 0:
                # Add contribution from this pulse
                start = max(0, i - pulse_width // 2)
                end = min(self.time_bins, i + pulse_width // 2)
                pulse_portion = pulse_shape[:end-start]
                measured[start:end] += clean_spectrum[i] * pulse_portion[:end-start]

        return measured

    def generate_random_chopper_pattern(
        self,
        duty_cycle: float = 0.1,
        pattern_type: str = 'random'
    ) -> np.ndarray:
        """
        Generate a chopper opening pattern.

        Args:
            duty_cycle: Fraction of time chopper is open
            pattern_type: 'random', 'pseudorandom', or 'adaptive'

        Returns:
            Binary array of chopper states
        """
        if pattern_type == 'random':
            return (np.random.rand(self.time_bins) < duty_cycle).astype(float)

        elif pattern_type == 'pseudorandom':
            # FOBI-like pseudorandom pattern with some structure
            n_slits = int(duty_cycle * self.time_bins / 10)
            pattern = np.zeros(self.time_bins)
            slit_positions = np.random.choice(self.time_bins, n_slits, replace=False)
            for pos in slit_positions:
                pattern[pos:min(pos+10, self.time_bins)] = 1
            return pattern

        elif pattern_type == 'adaptive':
            # Adaptive pattern focuses on regions with expected edges
            pattern = np.random.rand(self.time_bins) < duty_cycle * 0.5
            # Add extra sampling around wavelengths of interest (e.g., 2-4 Angstroms)
            mask = (self.wavelength_axis > 2) & (self.wavelength_axis < 4)
            pattern[mask] = (np.random.rand(mask.sum()) < duty_cycle * 2)
            return pattern.astype(float)

        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    def generate_training_pair(
        self,
        edge_positions: Optional[List[float]] = None,
        noise_level: float = 1000,
        duty_cycle: float = 0.1,
        pattern_type: str = 'random'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a single training example: (compressed measurement, target spectrum, chopper pattern).

        Args:
            edge_positions: Bragg edge positions (random if None)
            noise_level: Photon count level
            duty_cycle: Chopper duty cycle
            pattern_type: Type of chopper pattern

        Returns:
            (compressed_signal, target_spectrum, chopper_pattern)
        """
        # Generate random edge positions if not provided
        if edge_positions is None:
            n_edges = np.random.randint(1, 5)
            edge_positions = np.random.uniform(1.5, 4.0, n_edges)

        # Generate clean spectrum
        clean = self.generate_clean_spectrum(edge_positions)

        # Generate chopper pattern
        chopper = self.generate_random_chopper_pattern(duty_cycle, pattern_type)

        # Create overlapping measurement
        compressed = self.generate_overlapping_pulses(clean, chopper)

        # Add noise
        compressed = self.add_noise(compressed, noise_level)

        # Target is the clean spectrum (what we want to reconstruct)
        target = clean

        return compressed, target, chopper


def create_iron_bcc_spectrum(generator: BraggEdgeSignalGenerator) -> np.ndarray:
    """
    Create a realistic spectrum for BCC iron.

    Iron BCC Bragg edges:
    - (110): 2.027 Å
    - (200): 1.433 Å
    - (211): 1.170 Å
    """
    edge_positions = [2.027, 1.433, 1.170]
    edge_heights = [0.35, 0.25, 0.20]
    edge_widths = [0.04, 0.035, 0.03]

    return generator.generate_clean_spectrum(
        edge_positions,
        edge_heights,
        edge_widths,
        baseline=0.95,
        add_absorption=True
    )


def create_dataset(
    n_samples: int,
    generator: BraggEdgeSignalGenerator,
    duty_cycle: float = 0.1,
    pattern_type: str = 'random'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a full dataset for training.

    Returns:
        (compressed_signals, target_spectra, chopper_patterns) each of shape (n_samples, time_bins)
    """
    compressed = []
    targets = []
    choppers = []

    for _ in range(n_samples):
        comp, targ, chop = generator.generate_training_pair(
            noise_level=np.random.uniform(500, 5000),
            duty_cycle=duty_cycle,
            pattern_type=pattern_type
        )
        compressed.append(comp)
        targets.append(targ)
        choppers.append(chop)

    return (
        np.array(compressed),
        np.array(targets),
        np.array(choppers)
    )
