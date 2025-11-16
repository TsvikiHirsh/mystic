"""
Bragg Edge Deep Learning Reconstruction

ISACSNet-inspired neural network for reconstructing Bragg edge transmission
spectra from overlapping neutron time-of-flight pulses.
"""

__version__ = "0.1.0"
__author__ = "Bragg Edge DL Team"

from .models.isacs_net import ISACSNet
from .utils.signal_generator import BraggEdgeSignalGenerator, create_iron_bcc_spectrum

__all__ = [
    'ISACSNet',
    'BraggEdgeSignalGenerator',
    'create_iron_bcc_spectrum',
]
