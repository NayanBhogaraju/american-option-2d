#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.model import case_I
from core.payoffs import put_on_min_payoff, put_on_average_payoff
from core.pricer import AmericanOptionPricer2D


def domain_sensitivity(payoff_func, payoff_name, X0, Y0, N=256, J=256, M=100):
    model, _, _ = case_I(X0, Y0)

    half_widths = {
        'Smaller': 0.75,
        'Standard': 1.5,
        'Larger': 3.0,
    }

    print(f"\n{'─' * 55}")
    print(f"  Domain Sensitivity: {payoff_name}")
    print(f"  X0={X0}, Y0={Y0}, N={N}, J={J}, M={M}")
    print(f"{'─' * 55}")
    print(f"{'Domain':>12} {'Half-width':>12} {'Price':>14} {'Diff vs Std':>14}")
    print("-" * 55)

    prices = {}
    for label, hw in half_widths.items():
        scale = hw / 1.5
        N_scaled = int(N * scale)
        J_scaled = int(J * scale)
        N_scaled = max(N_scaled + N_scaled % 2, 4)
        J_scaled = max(J_scaled + J_scaled % 2, 4)

        pricer = AmericanOptionPricer2D(
            model, X0, Y0,
            N=N_scaled, J=J_scaled, M=M,
            domain_half_width_x=hw, domain_half_width_y=hw,
        )
        price = pricer.price(payoff_func)
        prices[label] = price

    std_price = prices['Standard']
    for label, price in prices.items():
        diff = abs(price - std_price)
        hw = half_widths[label]
        diff_str = f"{diff:.2e}" if label != 'Standard' else "—"
        print(f"{label:>12} {hw:>12.2f} {price:>14.6f} {diff_str:>14}")


def main():
    domain_sensitivity(put_on_min_payoff, "Put-on-Min", 90, 90)

    domain_sensitivity(put_on_average_payoff, "Put-on-Avg", 100, 100)


if __name__ == "__main__":
    main()
