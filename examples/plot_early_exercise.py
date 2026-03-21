#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from core.model import case_I
from core.payoffs import put_on_min_payoff, put_on_average_payoff
from core.pricer import AmericanOptionPricer2D


def plot_exercise_region(model, X0, Y0, payoff_func, payoff_name, N=256, J=256, M=100):
    pricer = AmericanOptionPricer2D(
        model, X0, Y0, N=N, J=J, M=M,
        domain_half_width_x=1.5, domain_half_width_y=1.5,
    )

    result = pricer.price(payoff_func, return_exercise_boundary=True)

    mid_idx = M // 2
    exercise = result['exercise_regions'][mid_idx]
    x_in = result['x_in']
    y_in = result['y_in']

    X_prices = np.exp(x_in)
    Y_prices = np.exp(y_in)
    XX, YY = np.meshgrid(X_prices, Y_prices, indexing='ij')

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    ax.contourf(XX, YY, exercise.astype(float), levels=[-0.5, 0.5, 1.5],
                colors=['#f0f4f8', '#4a90d9'], alpha=0.8)
    ax.contour(XX, YY, exercise.astype(float), levels=[0.5],
               colors='#1a3a5c', linewidths=1.5)

    ax.set_xlabel('X (Asset 1 price)', fontsize=12)
    ax.set_ylabel('Y (Asset 2 price)', fontsize=12)
    ax.set_title(f'Early Exercise Region at t = T/2\n{payoff_name}', fontsize=14)

    ax.text(0.7, 0.7, 'Continuation', transform=ax.transAxes,
            fontsize=14, fontstyle='italic', ha='center')
    ax.text(0.25, 0.25, 'Early\nexercise', transform=ax.transAxes,
            fontsize=14, fontstyle='italic', ha='center', fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    model, X0, Y0 = case_I(90.0, 90.0)

    fig1 = plot_exercise_region(
        model, X0, Y0, put_on_min_payoff,
        'American Put-on-the-Min (Case I)',
        N=256, J=256, M=100,
    )
    fig1.savefig('early_exercise_put_on_min.png', dpi=150, bbox_inches='tight')
    print("Saved: early_exercise_put_on_min.png")

    model, X0, Y0 = case_I(100.0, 100.0)

    fig2 = plot_exercise_region(
        model, X0, Y0, put_on_average_payoff,
        'American Put-on-the-Average (Case I)',
        N=256, J=256, M=100,
    )
    fig2.savefig('early_exercise_put_on_avg.png', dpi=150, bbox_inches='tight')
    print("Saved: early_exercise_put_on_avg.png")

    plt.show()


if __name__ == "__main__":
    main()
