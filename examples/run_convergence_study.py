#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.model import case_I
from core.payoffs import put_on_min_payoff, put_on_average_payoff
from core.pricer import run_convergence_study


def main():
    quick_levels = [
        {'N': 2**6, 'J': 2**6, 'M': 25},
        {'N': 2**7, 'J': 2**7, 'M': 50},
        {'N': 2**8, 'J': 2**8, 'M': 100},
        {'N': 2**9, 'J': 2**9, 'M': 200},
    ]

    print("=" * 65)
    print("Table 6.4: Put-on-the-min, Case I, X0=90, Y0=90")
    print("Reference price (operator splitting): 16.390")
    print("=" * 65)

    model, X0, Y0 = case_I(90.0, 90.0)
    results_min = run_convergence_study(
        model, X0, Y0,
        payoff_func=put_on_min_payoff,
        levels=quick_levels,
        domain_half_width=1.5,
    )

    print()
    print("=" * 65)
    print("Table 6.5: Put-on-the-average, Case I, X0=100, Y0=100")
    print("Reference price (operator splitting): 3.442")
    print("=" * 65)

    model, X0, Y0 = case_I(100.0, 100.0)
    results_avg = run_convergence_study(
        model, X0, Y0,
        payoff_func=put_on_average_payoff,
        levels=quick_levels,
        domain_half_width=1.5,
    )

    return results_min, results_avg


if __name__ == "__main__":
    main()
