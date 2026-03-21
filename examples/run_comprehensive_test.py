#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.model import case_I, case_II, case_III
from core.payoffs import put_on_min_payoff, put_on_average_payoff
from core.pricer import AmericanOptionPricer2D


REF_MIN = {
    'I': {
        (90, 90): 16.391, (90, 100): 13.999, (90, 110): 12.758,
        (100, 90): 13.021, (100, 100): 9.620, (100, 110): 7.877,
        (110, 90): 11.443, (110, 100): 7.227, (110, 110): 5.132,
    },
    'II': {
        (36, 36): 15.467, (36, 40): 14.564, (36, 44): 13.794,
        (40, 36): 14.092, (40, 40): 13.107, (40, 44): 12.263,
        (44, 36): 12.921, (44, 40): 11.877, (44, 44): 10.982,
    },
    'III': {
        (36, 36): 21.742, (36, 40): 20.908, (36, 44): 20.167,
        (40, 36): 21.272, (40, 40): 20.394, (40, 44): 19.611,
        (44, 36): 20.892, (44, 40): 19.983, (44, 44): 19.166,
    },
}

REF_AVG = {
    'I': {
        (90, 90): 10.003, (90, 100): 5.989, (90, 110): 3.441,
        (100, 90): 6.030, (100, 100): 3.442, (100, 110): 1.877,
        (110, 90): 3.491, (110, 100): 1.891, (110, 110): 0.993,
    },
    'II': {
        (36, 36): 5.406, (36, 40): 4.363, (36, 44): 3.547,
        (40, 36): 4.214, (40, 40): 3.339, (40, 44): 2.669,
        (44, 36): 3.225, (44, 40): 2.507, (44, 44): 1.969,
    },
    'III': {
        (36, 36): 12.466, (36, 40): 11.930, (36, 44): 11.440,
        (40, 36): 11.434, (40, 40): 10.943, (40, 44): 10.495,
        (44, 36): 10.493, (44, 40): 10.043, (44, 44): 9.633,
    },
}

DOMAIN_HW = {'I': 1.5, 'II': 3.0, 'III': 6.0}

CASE_FUNCS = {'I': case_I, 'II': case_II, 'III': case_III}


def run_table(case_name, payoff_func, payoff_label, ref_table, N=256, J=256, M=200):
    case_func = CASE_FUNCS[case_name]
    hw = DOMAIN_HW[case_name]
    prices = ref_table[case_name]

    print(f"\n{'─' * 60}")
    print(f"  {payoff_label} — Case {case_name}")
    print(f"{'─' * 60}")

    Y_vals = sorted(set(k[0] for k in prices.keys()))
    X_vals = sorted(set(k[1] for k in prices.keys()))

    print("{:>8}".format("Y0\\X0"), end="")
    for x in X_vals:
        print(f"{x:>12}", end="")
    print(f"  {'(ref)':>12}")

    for y in Y_vals:
        print(f"{y:>8.0f}", end="")
        for x in X_vals:
            model, _, _ = case_func(x, y)
            pricer = AmericanOptionPricer2D(
                model, x, y, N=N, J=J, M=M,
                domain_half_width_x=hw, domain_half_width_y=hw,
            )
            price = pricer.price(payoff_func)
            ref = prices.get((y, x), None)
            print(f"{price:>12.4f}", end="")
        ref_str = "  ref: " + ", ".join(f"{prices[(y, x)]:.3f}" for x in X_vals)
        print(ref_str)


def main():
    N, J, M = 256, 256, 100

    print("=" * 60)
    print("  Comprehensive Test: Tables 6.9 & 6.10")
    print(f"  Grid: N={N}, J={J}, M={M}")
    print("=" * 60)

    for case in ['I', 'II', 'III']:
        run_table(case, put_on_min_payoff, "Put-on-Min", REF_MIN, N, J, M)
        run_table(case, put_on_average_payoff, "Put-on-Avg", REF_AVG, N, J, M)


if __name__ == "__main__":
    main()
