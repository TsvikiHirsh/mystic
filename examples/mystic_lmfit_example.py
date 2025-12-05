"""
Example: Using lmfit API with mystic optimization engine

This example demonstrates how to use the mystic_lmfit module to perform
curve fitting with an lmfit-style API backed by mystic's powerful optimizers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mystic_lmfit import Model, create_params


def gaussian(x, amp, cen, wid):
    """Gaussian function for curve fitting."""
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))


def exponential_decay(t, amplitude, decay_rate, offset):
    """Exponential decay function."""
    return amplitude * np.exp(-decay_rate * t) + offset


def main():
    print("=" * 70)
    print("Mystic-LMfit Integration Examples")
    print("=" * 70)
    print()

    # Example 1: Gaussian fit with different solvers
    print("Example 1: Fitting a Gaussian with different mystic solvers")
    print("-" * 70)

    # Generate synthetic data
    x = np.linspace(0, 10, 100)
    true_amp, true_cen, true_wid = 8.5, 5.3, 1.2
    y_true_gauss = gaussian(x, true_amp, true_cen, true_wid)
    np.random.seed(42)
    noise = np.random.normal(0, 0.3, size=x.size)
    y_data_gauss = y_true_gauss + noise

    print(f"True parameters: amp={true_amp}, cen={true_cen}, wid={true_wid}")
    print()

    # Create model
    gmodel = Model(gaussian)

    # Create parameters with bounds
    params = gmodel.make_params(
        amp={'value': 5, 'min': 0, 'max': 15},
        cen={'value': 5, 'min': 0, 'max': 10},
        wid={'value': 1, 'min': 0.1, 'max': 5}
    )

    print("Initial parameters:")
    params.pretty_print()
    print()

    # Test different solvers
    methods = ['differential_evolution', 'nelder', 'powell']
    results = {}

    for method in methods:
        print(f"Fitting with method='{method}'...")
        result = gmodel.fit(y_data_gauss, params, x=x, method=method, maxiter=1000)
        results[method] = result

        print(f"  Best fit: amp={result.best_values['amp']:.4f}, "
              f"cen={result.best_values['cen']:.4f}, "
              f"wid={result.best_values['wid']:.4f}")
        print(f"  Chi-square: {result.chisqr:.6e}")
        print(f"  Reduced chi-square: {result.redchi:.6e}")
        print(f"  Function evaluations: {result.nfev}")
        print()

    # Example 2: Exponential decay with constraints
    print("\nExample 2: Exponential decay with parameter constraints")
    print("-" * 70)

    # Generate synthetic data
    t = np.linspace(0, 5, 50)
    true_A, true_k, true_offset = 10.0, 0.5, 2.0
    y_true_exp = exponential_decay(t, true_A, true_k, true_offset)
    np.random.seed(123)
    noise = np.random.normal(0, 0.2, size=t.size)
    y_data_exp = y_true_exp + noise

    print(f"True parameters: amplitude={true_A}, decay_rate={true_k}, offset={true_offset}")
    print()

    # Create model
    exp_model = Model(exponential_decay, independent_vars=['t'])

    # Create parameters with various constraints
    params = create_params(
        amplitude={'value': 8, 'min': 0, 'max': 20},
        decay_rate={'value': 1, 'min': 0.01, 'max': 2},
        offset={'value': 1, 'min': 0, 'max': 5}
    )

    print("Parameters with bounds:")
    params.pretty_print()
    print()

    # Fit with differential evolution (good for constrained problems)
    print("Fitting with differential_evolution solver...")
    result_exp = exp_model.fit(y_data_exp, params, t=t, method='differential_evolution')

    print(result_exp)

    # Example 3: Using fixed parameters
    print("\nExample 3: Fitting with some parameters fixed")
    print("-" * 70)

    # Reuse the exponential decay data
    params_fixed = create_params(
        amplitude={'value': 10, 'min': 0, 'max': 20},
        decay_rate={'value': 0.5, 'vary': False},  # Fix this parameter
        offset={'value': 2, 'min': 0, 'max': 5}
    )

    print("Parameters (decay_rate is fixed):")
    params_fixed.pretty_print()
    print()

    result_fixed = exp_model.fit(y_data_exp, params_fixed, t=t, method='nelder')

    print("Fit results with fixed decay_rate:")
    print(f"  amplitude={result_fixed.best_values['amplitude']:.4f}")
    print(f"  decay_rate={result_fixed.best_values['decay_rate']:.4f} (fixed)")
    print(f"  offset={result_fixed.best_values['offset']:.4f}")
    print(f"  Chi-square: {result_fixed.chisqr:.6e}")
    print()

    # Example 4: Comparing solver performance
    print("\nExample 4: Comparing solver accuracy")
    print("-" * 70)

    print(f"True values:      amp={true_amp:.4f}, cen={true_cen:.4f}, wid={true_wid:.4f}")
    print()
    for method, result in results.items():
        error = np.sqrt(
            (result.best_values['amp'] - true_amp)**2 +
            (result.best_values['cen'] - true_cen)**2 +
            (result.best_values['wid'] - true_wid)**2
        )
        print(f"{method:25s}: error={error:.6f}, nfev={result.nfev:4d}, chi2={result.chisqr:.4e}")
    print()

    # Create visualization
    print("\nCreating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Gaussian fits
    ax1.scatter(x, y_data_gauss, s=20, alpha=0.5, label='Data')
    ax1.plot(x, y_true_gauss, 'k--', linewidth=2, label='True model')

    for method, result in results.items():
        y_fit = gaussian(x, **result.best_values)
        ax1.plot(x, y_fit, linewidth=2, label=f'{method}')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Gaussian Fit Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Exponential decay fit
    ax2.scatter(t, y_data_exp, s=20, alpha=0.5, label='Data')
    ax2.plot(t, y_true_exp, 'k--', linewidth=2, label='True model')

    y_fit = exponential_decay(t, **result_exp.best_values)
    ax2.plot(t, y_fit, 'r-', linewidth=2, label='Fitted model')

    ax2.set_xlabel('t')
    ax2.set_ylabel('y')
    ax2.set_title('Exponential Decay Fit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = 'mystic_lmfit_example.png'
    plt.savefig(output_file, dpi=100)
    print(f"Figure saved to {output_file}")

    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
