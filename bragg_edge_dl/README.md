# Deep Learning Reconstruction for Bragg Edge ToF Spectroscopy

**ISACSNet-Inspired Neural Network for Overlapping Neutron Pulses**

This implementation adapts the ISACSNet architecture from Deng et al. (2025) to reconstruct clean Bragg edge transmission spectra from highly compressed, overlapping time-of-flight measurements using adaptive chopper sequences.

## Scientific Background

### The Problem

Traditional Bragg edge neutron ToF measurements face several challenges:
- **Low flux**: Single pulse per measurement limits neutron counts
- **Full spectrum acquisition**: Wasted beam time on non-informative wavelengths
- **Long measurement times**: Hours required for good statistics

### The Solution

This implementation demonstrates:
1. **Overlapping pulse measurements** using coded chopper patterns (FOBI-like approach)
2. **Deep learning reconstruction** to disentangle overlapping signals
3. **Adaptive measurement strategies** that focus on wavelengths of interest
4. **10-100× data compression** while preserving critical Bragg edge features

## Key Features

- **Learned measurement operators**: Replace random matrices with optimized convolutional filters
- **Multi-scale temporal processing**: Inception blocks capture features at different wavelength scales
- **LSTM temporal dependencies**: Handles sequential nature of ToF data
- **Self-attention refinement**: Identifies and enhances important spectral regions
- **Chopper pattern conditioning**: Adapts reconstruction to different measurement strategies

## Installation

```bash
# Install required packages
pip install numpy scipy matplotlib torch streamlit

# Or install from requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Synthetic Training Data and Train Model

```bash
cd bragg_edge_dl
python train.py
```

This will:
- Generate 5000 synthetic training samples
- Train the neural network for 100 epochs
- Save the best model to `./checkpoints/best_model.pth`
- Create training curves and reconstruction examples

### 2. Run Interactive Streamlit Demo

```bash
streamlit run app.py
```

Open your browser to the provided URL (typically `http://localhost:8501`) to:
- Adjust compression ratios and chopper patterns
- Generate different materials (Iron BCC, custom edges, random)
- See real-time reconstruction from overlapping pulses
- Analyze reconstruction quality metrics

### 3. Quick Demo Script

```python
from bragg_edge_dl.utils.signal_generator import BraggEdgeSignalGenerator, create_iron_bcc_spectrum
from bragg_edge_dl.models.isacs_net import ISACSNet
import torch
import numpy as np

# Create signal generator
generator = BraggEdgeSignalGenerator(time_bins=1024, flight_path=5.0)

# Generate Iron BCC spectrum
clean_spectrum = create_iron_bcc_spectrum(generator)

# Create adaptive chopper pattern (10% duty cycle)
chopper_pattern = generator.generate_random_chopper_pattern(0.1, 'adaptive')

# Simulate overlapping measurement
compressed_signal = generator.generate_overlapping_pulses(clean_spectrum, chopper_pattern)
compressed_signal = generator.add_noise(compressed_signal, photon_counts=5000)

# Load model and reconstruct
model = ISACSNet(signal_length=1024, compression_ratio=0.1)
# model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])

with torch.no_grad():
    compressed_tensor = torch.FloatTensor(compressed_signal).unsqueeze(0).unsqueeze(0)
    chopper_tensor = torch.FloatTensor(chopper_pattern).unsqueeze(0).unsqueeze(0)
    reconstructed = model(compressed_tensor, chopper_tensor)

print(f"Reconstruction shape: {reconstructed.shape}")
```

## Architecture Details

### ISACSNet Modules

1. **Random Projection Module**
   - 3 conv layers (kernels: 8, 8, 4)
   - Learns optimal compression patterns
   - Achieves specified compression ratio via pooling

2. **Dimension Raising Module**
   - Fully connected layers with LeakyReLU
   - Expands compressed representation

3. **Initial Reconstruction Module**
   - Modified Inception blocks (kernels: 3, 5, 7, 13)
   - Bidirectional LSTM (250 hidden units)
   - Skip connections

4. **Secondary Reconstruction Module**
   - Self-attention mechanism
   - Final refinement and edge enhancement

### Loss Function

Combined loss for edge-preserving reconstruction:
```
Loss = MSE(output, target) + α * MSE(∇output, ∇target)
```

Where:
- MSE term ensures overall accuracy
- Gradient term preserves sharp Bragg edges
- α = 0.5 balances the two objectives

## Expected Performance

Based on ISACSNet paper (Deng et al., 2025):

| Compression | ISACSNet PRD | Traditional CS PRD | Speedup |
|-------------|--------------|-------------------|---------|
| 10% (10×)   | ~5%          | 21-47%           | 1-4 orders |
| 5% (20×)    | ~7%          | >50%             | 1-4 orders |
| 1% (100×)   | ~9%          | >93% (fail)      | 1-4 orders |

**PRD**: Percentage Root-mean-square Difference (lower is better)

## Project Structure

```
bragg_edge_dl/
├── models/
│   └── isacs_net.py          # Neural network architecture
├── utils/
│   └── signal_generator.py   # Synthetic data generation
├── data/                      # (Generated data storage)
├── checkpoints/               # (Trained model checkpoints)
├── train.py                   # Training script
├── app.py                     # Streamlit interactive demo
└── README.md                  # This file
```

## Key Innovations

### Adaptation to Bragg Edge Problem

While the original ISACSNet addresses digital signal compression, this implementation extends it to:

1. **Physical measurement optimization**: Chopper patterns, not just digital sampling
2. **Adaptive strategies**: Pattern conditioning allows learning material-specific strategies
3. **Overlapping pulse reconstruction**: Handles FOBI-like coded aperture measurements
4. **Edge-preserving loss**: Maintains sharp Bragg edge features critical for crystallography

### Novel Contributions

- **Chopper pattern conditioning**: Network learns to adapt reconstruction based on measurement strategy
- **Adaptive sampling patterns**: Focuses measurement time on wavelengths of interest
- **Multi-material training**: Generalizes across different crystallographic structures
- **Edge-aware reconstruction**: Custom loss function preserves step-like Bragg edges

## Materials Examples

### Iron BCC (Body-Centered Cubic)
- (110) edge: 2.027 Å
- (200) edge: 1.433 Å
- (211) edge: 1.170 Å

### Custom Materials
Define your own Bragg edges for any crystalline material:
```python
edge_positions = [2.5, 1.8, 1.2]  # Angstroms
spectrum = generator.generate_clean_spectrum(edge_positions)
```

## Training Configuration

Default training hyperparameters (matching ISACSNet paper):
- **Learning rate**: 0.0005
- **Optimizer**: Adam
- **Batch size**: 16
- **Epochs**: 100-200
- **Training samples**: 5000-15000
- **Scheduler**: ReduceLROnPlateau

## Future Directions

This implementation demonstrates proof-of-concept for:

1. **Real-time adaptive measurement**: Update chopper patterns based on live reconstruction
2. **Bayesian experimental design**: Optimize next measurement based on current uncertainty
3. **Multi-pixel imaging**: Extend to 2D Bragg edge imaging with fast detectors
4. **Transfer learning**: Pre-train on simulations, fine-tune on real data
5. **Uncertainty quantification**: Ensemble methods or Bayesian neural networks

## References

**Primary Paper:**
Deng et al., "Deep learning-based compressed sampling reconstruction algorithm for digitizing intensive neutron ToF signals", Nuclear Science and Techniques, 2025
- DOI: 10.1007/s41365-025-01669-5
- URL: http://www.nst.sinap.ac.cn/article/doi/10.1007/s41365-025-01669-5

**Related Work:**
- FOBI (Frame Overlap Bragg edge Imaging): Busi et al., Nature, 2020
- Compressed sensing for neutron imaging
- Adaptive experimental design

## License

This implementation is provided for research and educational purposes.

## Citation

If you use this code in your research, please cite both the original ISACSNet paper and acknowledge this adaptation:

```bibtex
@article{deng2025isacs,
  title={Deep learning-based compressed sampling reconstruction algorithm for digitizing intensive neutron ToF signals},
  author={Deng et al.},
  journal={Nuclear Science and Techniques},
  year={2025},
  doi={10.1007/s41365-025-01669-5}
}
```

## Contact

For questions or collaboration opportunities related to this implementation,
please open an issue in the repository.

---

**Note**: This is a research demonstration adapting ISACSNet architecture to a novel
application domain (Bragg edge crystallography). Real experimental validation would
require integration with actual neutron ToF instrumentation and calibration with
measured reference spectra.
