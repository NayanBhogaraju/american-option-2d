import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import MertonJumpDiffusion2D
from core.model import case_I as model_case_I
from core.allocator import TwoAssetAllocator, crra_basket_utility, crra_min_utility


def main():
    model, X0, Y0 = model_case_I()
    print(f"Model: Case I  X0={X0}  Y0={Y0}  T={model.T}  r={model.r}")
    print(f"       sigma_x={model.sigma_x}  sigma_y={model.sigma_y}  "
          f"rho={model.rho}  lambda={model.lam}")

    N, J, M = 64, 64, 20
    n_pi = 11

    mu_x_real = 0.10
    mu_y_real = 0.11
    print(f"Real-world drifts: mu_x={mu_x_real}  mu_y={mu_y_real}  "
          f"(equity premium: {mu_x_real - model.r:.2f}, {mu_y_real - model.r:.2f})")

    print("\n" + "=" * 60)
    print("Part 1: Optimal value vs. risk-aversion gamma")
    print("=" * 60)
    print(f"{'gamma':>8}  {'Value V(X0,Y0)':>16}  {'pi_x* (center)':>16}  {'pi_y* (center)':>16}")
    print("-" * 62)

    gamma_values = [-2.0, -1.0, -0.5, 0.5]
    results_by_gamma = {}

    for gamma in gamma_values:
        terminal_util = crra_basket_utility(gamma, alpha=0.5)
        allocator = TwoAssetAllocator(
            model, X0, Y0, gamma=gamma,
            mu_x_real=mu_x_real, mu_y_real=mu_y_real,
            N=N, J=J, M=M, n_pi=n_pi,
            allow_short=False,
        )
        res = allocator.solve(
            terminal_util,
            return_grid_values=True,
            return_policy=True,
        )
        results_by_gamma[gamma] = res

        x_in = res['x']
        y_in = res['y']
        ix = np.argmin(np.abs(x_in - np.log(X0)))
        iy = np.argmin(np.abs(y_in - np.log(Y0)))
        px_center = res['pi_x'][ix, iy]
        py_center = res['pi_y'][ix, iy]

        print(f"{gamma:>8.2f}  {res['value']:>16.6f}  {px_center:>16.4f}  {py_center:>16.4f}")

    print("\n" + "=" * 60)
    print("Part 2: Optimal vs. fixed allocations (gamma = -1.0)")
    print("=" * 60)

    gamma = -1.0
    terminal_util = crra_basket_utility(gamma, alpha=0.5)

    allocator_opt = TwoAssetAllocator(
        model, X0, Y0, gamma=gamma,
        mu_x_real=mu_x_real, mu_y_real=mu_y_real,
        N=N, J=J, M=M, n_pi=n_pi, allow_short=False,
    )
    opt_res = allocator_opt.solve(terminal_util, return_policy=True)
    V_opt = opt_res['value']

    naive_configs = [
        ("All X  (pi_x=1, pi_y=0)", 1.0, 0.0),
        ("All Y  (pi_x=0, pi_y=1)", 0.0, 1.0),
        ("50/50  (pi_x=0.5, pi_y=0.5)", 0.5, 0.5),
        ("60/40  (pi_x=0.6, pi_y=0.4)", 0.6, 0.4),
        ("Cash   (pi_x=0, pi_y=0)", 0.0, 0.0),
    ]

    print(f"{'Strategy':>32}  {'Value V(X0,Y0)':>16}  {'vs. Optimal':>12}")
    print("-" * 65)
    print(f"{'Optimal (dynamic)':>32}  {V_opt:>16.6f}  {'---':>12}")

    for label, px, py in naive_configs:
        allocator_fixed = _build_single_policy_allocator(
            model, X0, Y0, gamma, mu_x_real, mu_y_real, N, J, M, px, py
        )
        V_fixed = allocator_fixed.solve(terminal_util)
        diff = V_fixed - V_opt
        print(f"{label:>32}  {V_fixed:>16.6f}  {diff:>+12.6f}")

    print("\n" + "=" * 60)
    print("Part 3: Generating plots")
    print("=" * 60)

    gamma = -1.0
    terminal_util = crra_basket_utility(gamma, alpha=0.5)

    allocator = TwoAssetAllocator(
        model, X0, Y0, gamma=gamma,
        mu_x_real=mu_x_real, mu_y_real=mu_y_real,
        N=N, J=J, M=M, n_pi=n_pi, allow_short=False,
    )
    res = allocator.solve(
        terminal_util,
        return_grid_values=True,
        return_policy=True,
    )

    x_in = res['x']
    y_in = res['y']
    V_grid = res['V']
    pi_x_grid = res['pi_x']
    pi_y_grid = res['pi_y']

    X_prices = np.exp(x_in)
    Y_prices = np.exp(y_in)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    Xm, Ym = np.meshgrid(X_prices, Y_prices, indexing='ij')
    cf = ax.contourf(Xm, Ym, V_grid, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='V(x, y, t=0)')
    ax.axvline(X0, color='red', linestyle='--', linewidth=1, label=f'X0={X0}')
    ax.axhline(Y0, color='white', linestyle='--', linewidth=1, label=f'Y0={Y0}')
    ax.set_xlabel('Asset X price')
    ax.set_ylabel('Asset Y price')
    ax.set_title(f'Value Function V(x,y,0)\ngamma={gamma}, M={M} rebalancing dates')
    ax.legend(fontsize=7)

    ax = axes[1]
    cf2 = ax.contourf(Xm, Ym, pi_x_grid, levels=np.linspace(0, 1, 12), cmap='RdYlGn')
    plt.colorbar(cf2, ax=ax, label='pi_x*')
    ax.axvline(X0, color='red', linestyle='--', linewidth=1)
    ax.axhline(Y0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Asset X price')
    ax.set_ylabel('Asset Y price')
    ax.set_title(f'Optimal allocation pi_x*(x,y)\n(fraction in Asset X)')

    ax = axes[2]
    total_risky = pi_x_grid + pi_y_grid
    cf3 = ax.contourf(Xm, Ym, total_risky, levels=np.linspace(0, 1, 12), cmap='plasma')
    plt.colorbar(cf3, ax=ax, label='pi_x* + pi_y*')
    ax.axvline(X0, color='white', linestyle='--', linewidth=1)
    ax.axhline(Y0, color='white', linestyle='--', linewidth=1)
    ax.set_xlabel('Asset X price')
    ax.set_ylabel('Asset Y price')
    ax.set_title(f'Total risky exposure pi_x*+pi_y*(x,y)\n(1 - cash fraction)')

    fig.suptitle(
        f'Two-Asset Portfolio Allocation  |  Case I  |  gamma={gamma}\n'
        f'Merton jump-diffusion, {M} rebalancing dates, {n_pi}x{n_pi} policy grid',
        fontsize=11
    )
    plt.tight_layout()

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'allocation_result.png'
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    print("\n" + "=" * 60)
    print("Part 4: Optimal pi_x* along X=X0 for different gamma")
    print("=" * 60)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for gamma in gamma_values:
        res = results_by_gamma[gamma]
        x_in = res['x']
        y_in = res['y']
        pi_x_g = res['pi_x']

        ix = np.argmin(np.abs(x_in - np.log(X0)))
        Y_slice = np.exp(y_in)
        pi_x_slice = pi_x_g[ix, :]
        ax1.plot(Y_slice, pi_x_slice, label=f'gamma={gamma}')

        iy = np.argmin(np.abs(y_in - np.log(Y0)))
        X_slice = np.exp(x_in)
        pi_x_xslice = pi_x_g[:, iy]
        ax2.plot(X_slice, pi_x_xslice, label=f'gamma={gamma}')

    ax1.axvline(Y0, color='gray', linestyle=':', linewidth=1)
    ax1.set_xlabel('Asset Y price')
    ax1.set_ylabel('Optimal pi_x*')
    ax1.set_title(f'pi_x*(X=X0, Y)  across gamma')
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)

    ax2.axvline(X0, color='gray', linestyle=':', linewidth=1)
    ax2.set_xlabel('Asset X price')
    ax2.set_ylabel('Optimal pi_x*')
    ax2.set_title(f'pi_x*(X, Y=Y0)  across gamma')
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)

    fig2.suptitle(
        'Sensitivity of optimal pi_x* to risk aversion  |  Case I  |  CRRA basket utility',
        fontsize=11
    )
    plt.tight_layout()

    out_path2 = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'allocation_policy_slices.png'
    )
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path2}")
    plt.close()

    print("\nDone.")


def _build_single_policy_allocator(
    model, X0, Y0, gamma, mu_x_real, mu_y_real, N, J, M, pi_x_fixed, pi_y_fixed
):
    class _FixedPolicyAllocator(TwoAssetAllocator):
        def _build_policy_grid(self):
            return np.array([[pi_x_fixed, pi_y_fixed]])

        def _precompute_policy_kernels(self):
            grid = self.grid
            gamma = self.gamma
            rf_factor = np.exp(self.model.r * grid.dtau)
            rn_correction = np.exp(self.model.r * grid.dtau)
            Dx, Dy = np.meshgrid(grid.x_ddag, grid.y_ddag, indexing='ij')
            R_p = (pi_x_fixed * np.exp(-Dx)
                   + pi_y_fixed * np.exp(-Dy)
                   + (1.0 - pi_x_fixed - pi_y_fixed) * rf_factor)
            g_pi = rn_correction * self._greens * np.maximum(R_p, 1e-30) ** gamma
            g_pi_padded = np.zeros((self._fft_nx, self._fft_ny))
            g_pi_padded[:self._n_disp_x, :self._n_disp_y] = g_pi
            return [np.fft.rfft2(g_pi_padded)]

    return _FixedPolicyAllocator(
        model, X0, Y0, gamma=gamma,
        mu_x_real=mu_x_real, mu_y_real=mu_y_real,
        N=N, J=J, M=M, n_pi=2, allow_short=False,
    )


if __name__ == '__main__':
    main()
