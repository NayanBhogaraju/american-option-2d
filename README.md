# Two-Asset American Option Pricing under Merton Jump-Diffusion

A Python implementation of the monotone integration scheme from:

> Zhou, H. & Dang, D.-M. (2025). "Numerical analysis of American option pricing
> in a two-asset jump-diffusion model." *Applied Numerical Mathematics*, 216, 98–126.

## Overview

This project prices **American options** on two correlated assets whose dynamics follow
a **Merton jump-diffusion model**. The numerical method uses:

1. An infinite series representation of the Green's function (each term non-negative)
2. 2-D convolution via composite trapezoidal rule (monotone by construction)
3. FFT-based circulant convolution for O(NJ log NJ) per-timestep cost
4. Explicit early exercise enforcement (max of continuation value and payoff)

The scheme is provably convergent to the viscosity solution of the variational inequality.

## Project Structure

```
american_option_2d/
├── core/
│   ├── __init__.py
│   ├── model.py          # Model parameters dataclass
│   ├── greens_function.py # Green's function series computation
│   ├── grid.py            # Spatial/temporal grid setup
│   ├── pricer.py          # Main pricing engine (Algorithm 4.2)
│   └── payoffs.py         # Payoff functions (put-on-min, put-on-avg, etc.)
├── tests/
│   ├── test_convergence.py # Convergence study (Tables 6.4, 6.5)
│   └── test_greens.py      # Green's function unit tests
├── examples/
│   ├── run_convergence_study.py    # Reproduce Tables 6.4 / 6.5
│   ├── run_comprehensive_test.py   # Reproduce Tables 6.9 / 6.10
│   ├── plot_early_exercise.py      # Reproduce Figs 6.1 / 6.2
│   └── run_domain_sensitivity.py   # Reproduce Tables 6.6 / 6.7
├── docs/
│   └── research_ideas.md  # Directions for extending this work
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install numpy scipy matplotlib
cd american_option_2d
python examples/run_convergence_study.py
```

## Key Parameters (Paper Table 6.1)

| Parameter | Case I | Case II | Case III |
|-----------|--------|---------|----------|
| σ_x       | 0.12   | 0.30    | 0.20     |
| σ_y       | 0.15   | 0.30    | 0.30     |
| ρ         | 0.30   | 0.50    | 0.70     |
| λ         | 0.60   | 2.00    | 8.00     |
| K         | 100    | 40      | 40       |
| T         | 1.0    | 0.5     | 1.0      |
| r         | 0.05   | 0.05    | 0.05     |

## License

Educational / research use. See the original paper for full attribution.
