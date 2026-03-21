#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.model import case_I, case_II, case_III
from core.greens_function import compute_greens_weights, truncation_bound
from core.grid import build_grid


def test_nonnegativity():
    for case_func, name in [(case_I, 'I'), (case_II, 'II'), (case_III, 'III')]:
        model, X0, Y0 = case_func()
        hw = {case_I: 1.5, case_II: 3.0, case_III: 6.0}[case_func]
        grid = build_grid(
            np.log(X0) - hw, np.log(X0) + hw,
            np.log(Y0) - hw, np.log(Y0) + hw,
            N=64, J=64, M=50, T=model.T,
        )
        weights = compute_greens_weights(model, grid.x_ddag, grid.y_ddag, grid.dtau)

        assert np.all(weights >= 0), f"Case {name}: found negative weights!"
        print(f"  Case {name}: all weights non-negative ✓ (min={weights.min():.2e})")


def test_integral_normalization():
    model, X0, Y0 = case_I()
    grid = build_grid(
        np.log(X0) - 1.5, np.log(X0) + 1.5,
        np.log(Y0) - 1.5, np.log(Y0) + 1.5,
        N=128, J=128, M=50, T=model.T,
    )
    weights = compute_greens_weights(model, grid.x_ddag, grid.y_ddag, grid.dtau)

    total = weights.sum()
    expected = np.exp(-model.r * grid.dtau)

    rel_error = abs(total - expected) / expected
    print(f"  Integral test: sum={total:.8f}, expected={expected:.8f}, rel_error={rel_error:.2e}")
    assert rel_error < 0.01, f"Normalization error too large: {rel_error:.4e}"
    print(f"  Normalization ✓")


def test_truncation_bound():
    model, _, _ = case_I()
    dtau = model.T / 50.0

    bounds = []
    for K in range(1, 20):
        b = truncation_bound(model, dtau, K)
        bounds.append(b)

    for i in range(1, len(bounds)):
        assert bounds[i] <= bounds[i - 1], \
            f"Truncation bound not decreasing: K={i+1}, {bounds[i]} > {bounds[i-1]}"

    print(f"  Truncation bounds decrease monotonically ✓")
    print(f"  K=1: {bounds[0]:.4e}, K=5: {bounds[4]:.4e}, K=10: {bounds[9]:.4e}")


def main():
    print("=" * 50)
    print("  Green's Function Unit Tests")
    print("=" * 50)

    print("\n1. Non-negativity test:")
    test_nonnegativity()

    print("\n2. Integral normalization test:")
    test_integral_normalization()

    print("\n3. Truncation bound test:")
    test_truncation_bound()

    print("\n" + "=" * 50)
    print("  All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
