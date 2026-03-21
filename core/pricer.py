import os
import sys
import numpy as np
from typing import Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.model import MertonJumpDiffusion2D
    from core.grid import Grid2D, build_grid
    from core.greens_function import compute_greens_weights
else:
    from .model import MertonJumpDiffusion2D
    from .grid import Grid2D, build_grid
    from .greens_function import compute_greens_weights


class AmericanOptionPricer2D:

    def __init__(
        self,
        model: MertonJumpDiffusion2D,
        X0: float,
        Y0: float,
        N: int = 256,
        J: int = 256,
        M: int = 50,
        domain_half_width_x: float = 1.5,
        domain_half_width_y: float = 1.5,
        series_tol: float = 1e-12,
    ):
        self.model = model
        self.X0 = X0
        self.Y0 = Y0
        self.N = N
        self.J = J
        self.M = M

        x0 = np.log(X0)
        y0 = np.log(Y0)
        x_min = x0 - domain_half_width_x
        x_max = x0 + domain_half_width_x
        y_min = y0 - domain_half_width_y
        y_max = y0 + domain_half_width_y

        self.grid = build_grid(x_min, x_max, y_min, y_max, N, J, M, model.T)

        self._n_dag_x = 2 * N + 1
        self._n_dag_y = 2 * J + 1
        self._n_disp_x = 3 * N - 1
        self._n_disp_y = 3 * J - 1

        self._fft_nx = self._next_pow2(self._n_dag_x + self._n_disp_x - 1)
        self._fft_ny = self._next_pow2(self._n_dag_y + self._n_disp_y - 1)

        g_weights = compute_greens_weights(
            model, self.grid.x_ddag, self.grid.y_ddag,
            self.grid.dtau, tol=series_tol,
        )

        g_padded = np.zeros((self._fft_nx, self._fft_ny))
        g_padded[:self._n_disp_x, :self._n_disp_y] = g_weights
        self._fft_g = np.fft.rfft2(g_padded)

    @staticmethod
    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p

    def _trapezoidal_weights_2d(self, nx: int, ny: int) -> np.ndarray:
        w = np.ones((nx, ny))
        w[0, :] *= 0.5
        w[-1, :] *= 0.5
        w[:, 0] *= 0.5
        w[:, -1] *= 0.5
        return w

    def price(
        self,
        payoff_func: Callable,
        return_grid_values: bool = False,
        return_exercise_boundary: bool = False,) -> float | dict:
        grid = self.grid
        model = self.model
        N, J, M = self.N, self.J, self.M
        dtau = grid.dtau

        X_dag, Y_dag = np.meshgrid(grid.x_dag, grid.y_dag, indexing='ij')
        payoff_dag = payoff_func(X_dag, Y_dag, model.K)
        v = payoff_dag.copy()

        X_in, Y_in = np.meshgrid(grid.x_in, grid.y_in, indexing='ij')
        payoff_in = payoff_func(X_in, Y_in, model.K)

        trap_w = self._trapezoidal_weights_2d(self._n_dag_x, self._n_dag_y)

        ix_start = N // 2 + 1
        ix_end = ix_start + (N - 1)
        jy_start = J // 2 + 1
        jy_end = jy_start + (J - 1)

        exercise_regions = [] if return_exercise_boundary else None

        out_x_start = 2 * N
        out_y_start = 2 * J
        n_in_x = N - 1
        n_in_y = J - 1

        for m in range(M):
            tau_next = (m + 1) * dtau

            signal = trap_w * v
            sig_padded = np.zeros((self._fft_nx, self._fft_ny))
            sig_padded[:self._n_dag_x, :self._n_dag_y] = signal

            fft_sig = np.fft.rfft2(sig_padded)
            conv_full = np.fft.irfft2(
                self._fft_g * fft_sig,
                s=(self._fft_nx, self._fft_ny),
            )

            u_interior = conv_full[
                out_x_start:out_x_start + n_in_x,
                out_y_start:out_y_start + n_in_y,
            ]

            v_interior = np.maximum(u_interior, payoff_in)

            if return_exercise_boundary:
                exercise = (payoff_in >= u_interior) & (payoff_in > 0)
                exercise_regions.append(exercise.copy())

            v[ix_start:ix_end, jy_start:jy_end] = v_interior

            discount = np.exp(-model.r * tau_next)
            v[:ix_start, :] = payoff_dag[:ix_start, :] * discount
            v[ix_end:, :] = payoff_dag[ix_end:, :] * discount
            v[:, :jy_start] = payoff_dag[:, :jy_start] * discount
            v[:, jy_end:] = payoff_dag[:, jy_end:] * discount

        x0_log = np.log(self.X0)
        y0_log = np.log(self.Y0)
        price = self._interpolate(grid.x_in, grid.y_in, v_interior, x0_log, y0_log)

        if return_grid_values or return_exercise_boundary:
            result = {'price': price}
            if return_grid_values:
                result['v'] = v_interior.copy()
                result['x'] = grid.x_in.copy()
                result['y'] = grid.y_in.copy()
            if return_exercise_boundary:
                result['exercise_regions'] = exercise_regions
                result['x_in'] = grid.x_in.copy()
                result['y_in'] = grid.y_in.copy()
            return result
        return price

    @staticmethod
    def _interpolate(x_grid, y_grid, values, x0, y0):
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (x_grid, y_grid), values,
            method='linear', bounds_error=False, fill_value=None,
        )
        return float(interp([x0, y0]).item())


def run_convergence_study(
    model: MertonJumpDiffusion2D,
    X0: float,
    Y0: float,
    payoff_func: Callable,
    levels: list[dict] | None = None,
    domain_half_width: float = 1.5,
    series_tol: float = 1e-12,
    verbose: bool = True,
) -> list[dict]:
    if levels is None:
        levels = [
            {'N': 2**8,  'J': 2**8,  'M': 50},
            {'N': 2**9,  'J': 2**9,  'M': 100},
            {'N': 2**10, 'J': 2**10, 'M': 200},
            {'N': 2**11, 'J': 2**11, 'M': 400},
        ]

    results = []
    prev_price = None

    if verbose:
        print(f"{'Level':>5} {'N':>6} {'J':>6} {'M':>5} "
              f"{'Price':>14} {'Change':>12} {'Ratio':>8}")
        print("-" * 65)

    for i, lev in enumerate(levels):
        pricer = AmericanOptionPricer2D(
            model, X0, Y0,
            N=lev['N'], J=lev['J'], M=lev['M'],
            domain_half_width_x=domain_half_width,
            domain_half_width_y=domain_half_width,
            series_tol=series_tol,
        )
        price = pricer.price(payoff_func)

        change = price - prev_price if prev_price is not None else None
        ratio = None
        if (len(results) >= 2 and results[-1]['change'] is not None
                and change and abs(change) > 1e-15):
            ratio = results[-1]['change'] / change

        results.append({
            'level': i, 'N': lev['N'], 'J': lev['J'], 'M': lev['M'],
            'price': price, 'change': change, 'ratio': ratio,
        })

        if verbose:
            ch = f"{change:.6e}" if change is not None else ""
            ra = f"{ratio:.2f}" if ratio is not None else ""
            print(f"{i:>5} {lev['N']:>6} {lev['J']:>6} {lev['M']:>5} "
                  f"{price:>14.6f} {ch:>12} {ra:>8}")

        prev_price = price

    return results
