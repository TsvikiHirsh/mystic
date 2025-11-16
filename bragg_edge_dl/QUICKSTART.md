# Quick Start Guide

## Installation Status

✅ numpy, scipy, matplotlib - INSTALLED
⏳ PyTorch, Streamlit - INSTALLING (in progress)

## Once Installation Completes

### 1. Verify Installation

```bash
python3 -c "import torch; import streamlit; print('All packages installed!')"
```

### 2. Run Quick Demo (No Training)

```bash
cd /home/user/mystic/bragg_edge_dl
python3 demo.py
```

**Output:**
- Generates Iron BCC spectrum with known Bragg edges
- Creates adaptive chopper pattern (10% duty cycle)
- Simulates overlapping pulses
- Performs reconstruction (untrained model for demo)
- Saves visualization to `demo_reconstruction.png`

### 3. Train the Model (Recommended)

```bash
cd /home/user/mystic/bragg_edge_dl
python3 train.py
```

**Training Process:**
- Generates 5000 synthetic training samples
- Trains for 100 epochs (~10-30 min on GPU, ~1-2 hours on CPU)
- Saves best model to `./checkpoints/best_model.pth`
- Creates training curves and examples

**Expected Output:**
```
Epoch 100/100
  Train Loss: 0.001234
  Val Loss:   0.001456
  Val PRD:    6.23%

Training complete! Best validation PRD: 5.87%
```

### 4. Launch Interactive Demo

```bash
cd /home/user/mystic/bragg_edge_dl
streamlit run app.py
```

**Features:**
- Interactive parameter adjustment
- Real-time reconstruction
- Multiple material types (Iron BCC, custom edges, random)
- Performance metrics visualization
- Educational content

Access at: `http://localhost:8501`

## Manual Package Installation (if needed)

If automatic installation didn't complete:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install streamlit
```

Or with GPU support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install streamlit
```

## Test Signal Generator (Works Now!)

The signal generator works with just numpy/scipy:

```python
from utils.signal_generator import BraggEdgeSignalGenerator, create_iron_bcc_spectrum
import numpy as np

# Create generator
generator = BraggEdgeSignalGenerator(time_bins=1024, flight_path=5.0)

# Generate Iron BCC spectrum
spectrum = create_iron_bcc_spectrum(generator)

# Create adaptive chopper
chopper = generator.generate_random_chopper_pattern(0.1, 'adaptive')

# Simulate overlapping measurement
compressed = generator.generate_overlapping_pulses(spectrum, chopper)

print(f"Generated spectrum with shape: {spectrum.shape}")
print(f"Chopper duty cycle: {np.sum(chopper>0.5)/len(chopper)*100:.1f}%")
print("Success!")
```

## Example Usage

### Generate Custom Material

```python
from utils.signal_generator import BraggEdgeSignalGenerator

generator = BraggEdgeSignalGenerator(time_bins=1024)

# Your custom material edges (in Angstroms)
my_edges = [2.5, 1.8, 1.2]

spectrum = generator.generate_clean_spectrum(my_edges)
print(f"Generated spectrum for custom material")
```

### Reconstruct with Trained Model

```python
import torch
from models.isacs_net import ISACSNet

# Load model
model = ISACSNet(signal_length=1024, compression_ratio=0.1)
checkpoint = torch.load('./checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Reconstruct
with torch.no_grad():
    compressed_tensor = torch.FloatTensor(compressed).unsqueeze(0).unsqueeze(0)
    chopper_tensor = torch.FloatTensor(chopper).unsqueeze(0).unsqueeze(0)

    reconstructed = model(compressed_tensor, chopper_tensor)

print(f"Reconstructed spectrum with shape: {reconstructed.shape}")
```

## Troubleshooting

### "No module named 'torch'"

**Solution:** Wait for installation to complete, or install manually:
```bash
pip install torch
```

### "No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit
```

### Training is slow

**Solutions:**
- Use GPU if available (automatic detection)
- Reduce batch size or model size
- Reduce number of training samples for quick test

### Model not in checkpoints/

**Solution:** Run training first:
```bash
python3 train.py
```

## Project Files

```
bragg_edge_dl/
├── models/isacs_net.py       # Neural network
├── utils/signal_generator.py  # Data generation
├── train.py                   # Training script
├── demo.py                    # Quick demo
├── app.py                     # Streamlit app
├── README.md                  # Full documentation
├── USAGE_GUIDE.md            # Detailed examples
└── QUICKSTART.md             # This file
```

## Performance Targets

After training, expect:
- **PRD < 10%** at 10% compression
- **Correlation > 0.99** with ground truth
- **Sharp Bragg edge preservation**
- **Reconstruction time: ~10ms** per spectrum

## Next Steps

1. ✅ Installation complete
2. ✅ Run demo.py
3. ✅ Train model (train.py)
4. ✅ Launch Streamlit app
5. ✅ Read USAGE_GUIDE.md for advanced features

## Questions?

- Check README.md for overview
- Check USAGE_GUIDE.md for examples
- Check BRAGG_EDGE_DL_SUMMARY.md for technical details

---

**Ready to go!** Start with `python3 demo.py`
