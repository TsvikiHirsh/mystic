# Usage Guide: Bragg Edge Deep Learning Reconstruction

## Quick Start Examples

### Example 1: Generate and Visualize Synthetic Data

```python
import numpy as np
import matplotlib.pyplot as plt
from utils.signal_generator import BraggEdgeSignalGenerator, create_iron_bcc_spectrum

# Create signal generator for 5m flight path
generator = BraggEdgeSignalGenerator(
    time_bins=1024,
    time_range=(0, 10000),  # μs
    flight_path=5.0  # meters
)

# Generate Iron BCC spectrum with known Bragg edges
clean_spectrum = create_iron_bcc_spectrum(generator)

# Create adaptive chopper pattern (10% duty cycle)
chopper = generator.generate_random_chopper_pattern(
    duty_cycle=0.1,
    pattern_type='adaptive'
)

# Simulate overlapping measurement
compressed = generator.generate_overlapping_pulses(
    clean_spectrum, chopper, pulse_width=50
)

# Add realistic Poisson noise
noisy = generator.add_noise(compressed, photon_counts=5000)

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

axes[0].plot(generator.time_axis, clean_spectrum, 'g-', lw=2)
axes[0].set_title('Clean Bragg Edge Spectrum')
axes[0].set_ylabel('Transmission')

axes[1].plot(generator.time_axis, compressed, 'b-')
axes[1].fill_between(generator.time_axis, 0, compressed.max(),
                     where=chopper > 0.5, alpha=0.2, color='green')
axes[1].set_title('Compressed Overlapping Measurement')
axes[1].set_ylabel('Intensity')

axes[2].plot(generator.time_axis, noisy, 'r-', alpha=0.7)
axes[2].set_title('Noisy Compressed Measurement')
axes[2].set_xlabel('Time of Flight (μs)')
axes[2].set_ylabel('Intensity')

plt.tight_layout()
plt.show()
```

### Example 2: Train the Neural Network

```python
from train import train_model

# Train with default parameters
model, train_losses, val_losses, val_prds = train_model(
    signal_length=1024,
    compression_ratio=0.1,  # 10% sampling
    n_train=5000,
    n_val=500,
    batch_size=16,
    epochs=100,
    learning_rate=0.0005,
    device='auto',  # Uses CUDA if available
    save_dir='./checkpoints'
)

# Best model saved to ./checkpoints/best_model.pth
print(f"Best validation PRD: {min(val_prds):.2f}%")
```

### Example 3: Use Trained Model for Reconstruction

```python
import torch
from models.isacs_net import ISACSNet
from utils.signal_generator import BraggEdgeSignalGenerator
import numpy as np

# Load trained model
model = ISACSNet(signal_length=1024, compression_ratio=0.1)
checkpoint = torch.load('./checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate test signal
generator = BraggEdgeSignalGenerator(time_bins=1024)
compressed, target, chopper = generator.generate_training_pair(
    edge_positions=[2.0, 1.5, 1.2],  # Custom material
    noise_level=5000,
    duty_cycle=0.1
)

# Reconstruct
with torch.no_grad():
    compressed_tensor = torch.FloatTensor(compressed).unsqueeze(0).unsqueeze(0)
    chopper_tensor = torch.FloatTensor(chopper).unsqueeze(0).unsqueeze(0)

    reconstructed = model(compressed_tensor, chopper_tensor)
    reconstructed_np = reconstructed.cpu().numpy()[0, 0]

# Calculate reconstruction quality
prd = np.sqrt(np.sum((reconstructed_np - target)**2) / np.sum(target**2)) * 100
print(f"Reconstruction PRD: {prd:.2f}%")
```

### Example 4: Batch Process Multiple Measurements

```python
from utils.signal_generator import create_dataset
from models.isacs_net import ISACSNet
import torch

# Generate batch of test data
generator = BraggEdgeSignalGenerator(time_bins=1024)
compressed_signals, target_spectra, chopper_patterns = create_dataset(
    n_samples=100,
    generator=generator,
    duty_cycle=0.1,
    pattern_type='adaptive'
)

# Load model
model = ISACSNet(signal_length=1024, compression_ratio=0.1)
checkpoint = torch.load('./checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Batch reconstruction
batch_size = 16
all_reconstructed = []

with torch.no_grad():
    for i in range(0, len(compressed_signals), batch_size):
        batch_compressed = torch.FloatTensor(
            compressed_signals[i:i+batch_size]
        ).unsqueeze(1)
        batch_chopper = torch.FloatTensor(
            chopper_patterns[i:i+batch_size]
        ).unsqueeze(1)

        batch_output = model(batch_compressed, batch_chopper)
        all_reconstructed.append(batch_output.cpu().numpy())

reconstructed_spectra = np.concatenate(all_reconstructed, axis=0)[:, 0, :]
print(f"Reconstructed {len(reconstructed_spectra)} spectra")

# Compute average PRD
prds = []
for i in range(len(reconstructed_spectra)):
    prd = np.sqrt(np.sum((reconstructed_spectra[i] - target_spectra[i])**2) /
                  np.sum(target_spectra[i]**2)) * 100
    prds.append(prd)

print(f"Average PRD: {np.mean(prds):.2f}% ± {np.std(prds):.2f}%")
```

## Advanced Usage

### Custom Materials

Define custom Bragg edge positions for any crystalline material:

```python
# Example: Aluminum FCC
# (111) at 2.338 Å, (200) at 2.024 Å, (220) at 1.431 Å
aluminum_edges = [2.338, 2.024, 1.431]
aluminum_heights = [0.30, 0.25, 0.20]
aluminum_widths = [0.04, 0.035, 0.03]

spectrum = generator.generate_clean_spectrum(
    edge_positions=aluminum_edges,
    edge_heights=aluminum_heights,
    edge_widths=aluminum_widths,
    baseline=0.95,
    add_absorption=True
)
```

### Experiment with Different Chopper Patterns

```python
# Random pattern
random_chopper = generator.generate_random_chopper_pattern(
    duty_cycle=0.1,
    pattern_type='random'
)

# Pseudorandom pattern (FOBI-like)
fobi_chopper = generator.generate_random_chopper_pattern(
    duty_cycle=0.1,
    pattern_type='pseudorandom'
)

# Adaptive pattern (focuses on specific wavelengths)
adaptive_chopper = generator.generate_random_chopper_pattern(
    duty_cycle=0.1,
    pattern_type='adaptive'
)

# Compare reconstruction quality with each pattern
```

### Adjust Network Architecture

```python
from models.isacs_net import ISACSNet

# Smaller model for faster training
model_small = ISACSNet(
    signal_length=512,  # Fewer time bins
    compression_ratio=0.1,
    channels=32,  # Fewer channels
    use_chopper_conditioning=True
)

# Larger model for better accuracy
model_large = ISACSNet(
    signal_length=2048,  # More time bins
    compression_ratio=0.05,  # Higher compression
    channels=128,  # More channels
    use_chopper_conditioning=True
)

# Without chopper conditioning (simpler model)
model_no_cond = ISACSNet(
    signal_length=1024,
    compression_ratio=0.1,
    use_chopper_conditioning=False
)
```

### Transfer Learning from Pre-trained Model

```python
# Load pre-trained model
pretrained = ISACSNet(signal_length=1024, compression_ratio=0.1)
checkpoint = torch.load('./checkpoints/best_model.pth')
pretrained.load_state_dict(checkpoint['model_state_dict'])

# Create new model for different compression ratio
new_model = ISACSNet(signal_length=1024, compression_ratio=0.05)

# Transfer weights from compatible layers
new_model.initial_reconstruction.load_state_dict(
    pretrained.initial_reconstruction.state_dict()
)
new_model.secondary_reconstruction.load_state_dict(
    pretrained.secondary_reconstruction.state_dict()
)

# Fine-tune with new compression ratio
# ... training code ...
```

## Performance Optimization

### GPU Acceleration

```python
import torch

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Move model to GPU
model = ISACSNet(...).to(device)

# Move data to GPU
data = data.to(device)
```

### Mixed Precision Training (for faster training on modern GPUs)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    data, target = batch['compressed'].cuda(), batch['target'].cuda()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Parallel Data Loading

```python
from torch.utils.data import DataLoader

# Use multiple workers for faster data loading
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel loading
    pin_memory=True  # Faster CPU->GPU transfer
)
```

## Troubleshooting

### Issue: Poor Reconstruction Quality

**Solutions:**
1. Train longer (increase epochs to 200-300)
2. Generate more training data (n_train=10000-15000)
3. Adjust learning rate (try 0.0001-0.001)
4. Increase model capacity (more channels)
5. Check that chopper duty cycle matches training

### Issue: Model Not Converging

**Solutions:**
1. Reduce learning rate
2. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. Check for NaN values in training data
4. Reduce batch size for more frequent updates

### Issue: Out of Memory (GPU)

**Solutions:**
1. Reduce batch size
2. Reduce model size (fewer channels, shorter signals)
3. Use gradient checkpointing
4. Clear cache: `torch.cuda.empty_cache()`

### Issue: Slow Training

**Solutions:**
1. Use GPU instead of CPU
2. Enable mixed precision training
3. Increase batch size (if memory allows)
4. Use parallel data loading
5. Reduce signal length or model size

## Integration with Real Experiments

### Adapting to Real Data

```python
# Load experimental ToF data
experimental_tof = np.loadtxt('path/to/data.txt')

# Resample to match model input size
from scipy.interpolate import interp1d
f = interp1d(np.arange(len(experimental_tof)), experimental_tof)
resampled = f(np.linspace(0, len(experimental_tof)-1, 1024))

# Normalize to expected range
resampled = resampled / resampled.max()

# Reconstruct
with torch.no_grad():
    input_tensor = torch.FloatTensor(resampled).unsqueeze(0).unsqueeze(0)
    chopper_tensor = torch.FloatTensor(chopper_pattern).unsqueeze(0).unsqueeze(0)
    reconstructed = model(input_tensor, chopper_tensor)
```

### Calibration with Reference Materials

```python
# Use known reference materials to calibrate
reference_materials = {
    'iron': [2.027, 1.433, 1.170],  # Known edges in Angstroms
    'aluminum': [2.338, 2.024, 1.431],
    # ... more materials
}

# Fine-tune model on reference measurements
# ... training code with real data ...
```

## Citation and References

When using this code in research, please cite:

```bibtex
@article{deng2025isacs,
  title={Deep learning-based compressed sampling reconstruction algorithm
         for digitizing intensive neutron ToF signals},
  author={Deng et al.},
  journal={Nuclear Science and Techniques},
  year={2025},
  doi={10.1007/s41365-025-01669-5}
}
```

## Support

For issues, questions, or contributions:
- Check the README.md for general information
- Review this usage guide for examples
- Open an issue in the repository
- Contact the development team

---

**Last Updated:** 2025-11-16
