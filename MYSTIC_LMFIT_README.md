# Mystic-LMfit Integration

An **lmfit-compatible API** that uses **mystic** as the underlying optimization engine.

## Overview

This integration provides users familiar with lmfit's API with access to mystic's powerful optimization solvers and constraint handling capabilities. You get the same intuitive interface you know from lmfit, backed by mystic's robust global optimization algorithms.

## Key Features

- **Familiar lmfit API**: Use `Model`, `Parameters`, and `fit()` just like in lmfit
- **Mystic solvers**: Access to differential evolution, Nelder-Mead, Powell, and ensemble solvers
- **Advanced constraints**: Leverage mystic's sophisticated constraint handling
- **Parameter bounds**: Simple min/max bounds on parameters
- **Fixed parameters**: Easy parameter fixing via `vary=False`
- **Multiple solvers**: Switch between solvers using the `method` argument

## Installation

The integration is provided as a standalone module `mystic_lmfit.py` in the mystic repository.

```bash
# Install mystic with dependencies
pip install mystic

# The mystic_lmfit module is ready to use!
```

## Quick Start

```python
import numpy as np
from mystic_lmfit import Model, create_params

# Define your model function
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

# Generate some data
x = np.linspace(0, 10, 100)
y_data = gaussian(x, 8.5, 5.3, 1.2) + np.random.normal(0, 0.3, size=100)

# Create a Model
model = Model(gaussian)

# Create parameters with bounds
params = model.make_params(
    amp={'value': 5, 'min': 0, 'max': 15},
    cen={'value': 5, 'min': 0, 'max': 10},
    wid={'value': 1, 'min': 0.1, 'max': 5}
)

# Fit using mystic's differential evolution solver
result = model.fit(y_data, params, x=x, method='differential_evolution')

# View results
print(result)
print(f"Best fit: amp={result.best_values['amp']:.4f}")
print(f"Chi-square: {result.chisqr:.6e}")
```

## Available Solvers

Specify the solver using the `method` argument in `fit()` or when creating the `Model`:

### Local Optimizers (Fast, gradient-free)

- **`'nelder'`** or **`'neldermead'`**: Nelder-Mead Simplex algorithm
  - Best for: Smooth, well-behaved objective functions
  - Speed: Very fast
  - Robustness: May get stuck in local minima

- **`'powell'`**: Powell's Directional Search
  - Best for: Problems where derivatives are expensive or unavailable
  - Speed: Fast
  - Robustness: Good for local optimization

### Global Optimizers (Robust, thorough search)

- **`'differential_evolution'`** or **`'de'`**: Differential Evolution (default)
  - Best for: Complex landscapes, global optimization, constrained problems
  - Speed: Slower but thorough
  - Robustness: Excellent, less likely to get stuck in local minima
  - Population-based stochastic search

### Ensemble Solvers (Experimental)

- **`'buckshot'`**: Multiple random start points
- **`'lattice'`**: Grid-based ensemble solver

## Parameter Configuration

### Method 1: Using `make_params()`

```python
# Simple values
params = model.make_params(amp=5, cen=5, wid=1)

# With bounds and constraints
params = model.make_params(
    amp={'value': 5, 'min': 0, 'max': 15},
    cen={'value': 5, 'min': 0, 'max': 10, 'vary': True},
    wid={'value': 1, 'min': 0.1, 'max': 5, 'vary': True}
)
```

### Method 2: Using `create_params()`

```python
from mystic_lmfit import create_params

params = create_params(
    amplitude={'value': 10, 'min': 0, 'max': 20},
    decay_rate={'value': 0.5, 'min': 0.01, 'max': 2},
    offset={'value': 2, 'min': 0, 'max': 5}
)
```

### Method 3: Manual Parameter Creation

```python
from mystic_lmfit import Parameters

params = Parameters()
params.add('amp', value=5, min=0, max=15)
params.add('cen', value=5, min=0, max=10)
params.add('wid', value=1, min=0.1, max=5, vary=True)
```

## Fixing Parameters

Fix parameters by setting `vary=False`:

```python
params = create_params(
    amplitude={'value': 10, 'min': 0, 'max': 20},
    decay_rate={'value': 0.5, 'vary': False},  # Fixed parameter
    offset={'value': 2, 'min': 0, 'max': 5}
)

result = model.fit(data, params, x=x, method='nelder')
```

## Independent Variables

Specify independent variables when creating the model:

```python
# For functions with custom independent variable names
def exponential_decay(t, amplitude, decay_rate, offset):
    return amplitude * np.exp(-decay_rate * t) + offset

model = Model(exponential_decay, independent_vars=['t'])

# Pass the independent variable when fitting
result = model.fit(data, params, t=time_points)
```

## Fit Options

The `fit()` method accepts several options:

```python
result = model.fit(
    data,                           # Data to fit
    params,                         # Parameters object
    method='differential_evolution', # Solver to use
    x=x,                            # Independent variable(s)
    ftol=1e-6,                      # Function tolerance
    xtol=1e-6,                      # Parameter tolerance
    maxiter=1000,                   # Maximum iterations
    maxfev=None,                    # Maximum function evaluations
    weights=None,                   # Weights for each data point
    nan_policy='raise'              # How to handle NaN ('raise', 'propagate', 'omit')
)
```

## Result Object

The `ModelResult` object contains:

```python
result.params          # Updated Parameters object
result.best_values     # Dict of parameter names to best-fit values
result.chisqr          # Chi-square statistic
result.redchi          # Reduced chi-square
result.residual        # Residuals array
result.nfev            # Number of function evaluations
result.nvarys          # Number of varying parameters
result.ndata           # Number of data points
result.success         # Boolean: did the fit succeed?
result.message         # Message about the fit
```

## Example: Full Workflow

```python
import numpy as np
from mystic_lmfit import Model, create_params

# 1. Define model
def exponential_decay(t, amplitude, decay_rate, offset):
    return amplitude * np.exp(-decay_rate * t) + offset

# 2. Generate synthetic data
t = np.linspace(0, 5, 50)
y_true = exponential_decay(t, 10.0, 0.5, 2.0)
y_data = y_true + np.random.normal(0, 0.2, size=50)

# 3. Create model with custom independent variable
model = Model(exponential_decay, independent_vars=['t'])

# 4. Set up parameters with bounds
params = create_params(
    amplitude={'value': 8, 'min': 0, 'max': 20},
    decay_rate={'value': 1, 'min': 0.01, 'max': 2},
    offset={'value': 1, 'min': 0, 'max': 5}
)

# 5. Fit using differential evolution (good for global optimization)
result = model.fit(y_data, params, t=t, method='differential_evolution')

# 6. Display results
print(result)
print(f"\nBest fit:")
print(f"  amplitude = {result.best_values['amplitude']:.4f}")
print(f"  decay_rate = {result.best_values['decay_rate']:.4f}")
print(f"  offset = {result.best_values['offset']:.4f}")
print(f"\nGoodness of fit:")
print(f"  Chi-square = {result.chisqr:.6e}")
print(f"  Reduced chi-square = {result.redchi:.6e}")
```

## Comparison with Standard lmfit

### Similarities

- Same `Model` and `Parameters` API
- Same `fit()` interface
- Compatible parameter specifications
- Similar result object structure

### Differences

| Feature | lmfit | mystic_lmfit |
|---------|-------|-------------|
| Default solver | Levenberg-Marquardt | Differential Evolution |
| Optimization backend | scipy.optimize | mystic |
| Global optimization | Limited | Excellent |
| Constraint handling | Good | Advanced (via mystic) |
| Speed (local) | Very fast | Moderate |
| Speed (global) | N/A | Good |
| Parameter correlations | Yes | No (yet) |
| Confidence intervals | Yes | No (yet) |

## When to Use Mystic-LMfit

**Use mystic_lmfit when you need:**

- Global optimization (finding the global minimum)
- Robust optimization for complex objective functions
- Advanced constraint handling
- Population-based search algorithms
- An lmfit-style interface with mystic's power

**Use standard lmfit when you need:**

- Very fast local optimization (Levenberg-Marquardt)
- Parameter uncertainties and correlations
- Confidence intervals
- Maximum compatibility with lmfit ecosystem

## Examples

See `examples/mystic_lmfit_example.py` for comprehensive examples including:

1. Gaussian fitting with multiple solvers
2. Exponential decay with parameter constraints
3. Fixed parameters
4. Solver comparison

Run the example:

```bash
python examples/mystic_lmfit_example.py
```

## Advanced: Using Mystic Constraints

While this integration provides basic bounds via the lmfit API, you can access mystic's advanced constraint features by directly modifying the solver. This is an advanced use case - see the mystic documentation for details.

## Implementation Details

- **Module**: `mystic_lmfit.py`
- **Classes**: `Model`, `Parameters`, `Parameter`, `ModelResult`
- **Functions**: `create_params()`
- **Dependencies**: mystic, numpy

The implementation translates between lmfit's API and mystic's optimization framework:

1. lmfit `Parameters` → mystic initial points + bounds
2. lmfit `Model` function → mystic cost function (sum of squared residuals)
3. `method` argument → mystic solver selection
4. mystic solver results → lmfit-compatible `ModelResult`

## API Reference

### Model

```python
Model(func, independent_vars=None, method='differential_evolution', **kwargs)
```

Create a model from a function.

**Methods:**
- `make_params(**kwargs)`: Create a Parameters object
- `fit(data, params, method, ...)`: Fit the model to data

### Parameters

```python
Parameters()
```

An ordered dictionary of Parameter objects.

**Methods:**
- `add(name, value, vary, min, max, ...)`: Add a parameter
- `valuesdict()`: Get parameter values as a dictionary
- `pretty_print()`: Print parameters nicely

### Parameter

```python
Parameter(name, value=None, vary=True, min=-inf, max=inf, ...)
```

A single parameter with value, bounds, and metadata.

### create_params

```python
create_params(**kwargs)
```

Convenience function to create a Parameters object from keyword arguments.

## Contributing

This integration is part of the mystic project. Contributions are welcome!

- **Issues**: Report bugs or request features on the mystic GitHub
- **Pull requests**: Submit improvements via GitHub
- **Documentation**: Help improve these docs

## License

This integration follows mystic's license (3-clause BSD).

## References

- **mystic**: https://github.com/uqfoundation/mystic
- **lmfit**: https://lmfit.github.io/lmfit-py/
- **Example**: `examples/mystic_lmfit_example.py`

## Acknowledgments

- **lmfit authors**: For the excellent API design
- **mystic authors**: For the powerful optimization framework
- **Users**: For valuable feedback and contributions
