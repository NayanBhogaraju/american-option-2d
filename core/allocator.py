import os
import sys
import numpy as np
from typing import Callable

try:
    import scipy.fft as _fft
except ImportError:          # fallback: numpy.fft is API-compatible
    import numpy.fft as _fft  # type: ignore

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.model import MertonJumpDiffusion2D
    from core.grid import Grid2D, build_grid
    from core.greens_function import compute_greens_weights
else:
    from .model import MertonJumpDiffusion2D
    from .grid import Grid2D, build_grid
    from .greens_function import compute_greens_weights


def crra_basket_utility(gamma: float, alpha: float = 0.5, growth_tilt: float = 0.0):
    """
    growth_tilt in [0, 1]:
      0.0 → pure CRRA (risk-aversion = gamma)
      1.0 → log-utility / Kelly criterion (maximises long-run geometric growth)
    Intermediate values interpolate gamma_eff = gamma * (1 - tilt).
    """
    gamma_eff = gamma * (1.0 - growth_tilt)
    if abs(gamma_eff) < 1e-6:
        return log_basket_utility(alpha)

    def _util(x, y):
        W = np.maximum(alpha * np.exp(x) + (1.0 - alpha) * np.exp(y), 1e-30)
        return (W ** gamma_eff) / gamma_eff

    return _util


def log_basket_utility(alpha: float = 0.5):
    def _util(x, y):
        W = alpha * np.exp(x) + (1.0 - alpha) * np.exp(y)
        return np.log(np.maximum(W, 1e-30))

    return _util


def crra_min_utility(gamma: float):
    if abs(gamma) < 1e-10:
        raise ValueError("Use log_basket_utility for log utility.")

    def _util(x, y):
        W = np.minimum(np.exp(x), np.exp(y))
        return (np.maximum(W, 1e-30) ** gamma) / gamma

    return _util


class TwoAssetAllocator:

    def __init__(
        self,
        model: MertonJumpDiffusion2D,
        X0: float,
        Y0: float,
        gamma: float,
        mu_x_real: float | None = None,
        mu_y_real: float | None = None,
        N: int = 64,
        J: int = 64,
        M: int = 20,
        n_pi: int = 11,
        allow_short: bool = False,
        domain_half_width_x: float = 1.5,
        domain_half_width_y: float = 1.5,
        series_tol: float = 1e-12,
    ):
        if gamma >= 1.0:
            raise ValueError(f"CRRA gamma={gamma} must be < 1.")
        if abs(gamma) < 1e-10:
            raise ValueError(
                "gamma=0 (log utility) not directly supported. "
                "Use log_basket_utility as the terminal condition with gamma close to 0."
            )

        self.model = model
        self.X0 = X0
        self.Y0 = Y0
        self.gamma = gamma
        self.mu_x_real = mu_x_real
        self.mu_y_real = mu_y_real
        self.N = N
        self.J = J
        self.M = M
        self.n_pi = n_pi
        self.allow_short = allow_short

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

        greens_model = model
        if mu_x_real is not None or mu_y_real is not None:
            greens_model = self._make_real_world_model(model, mu_x_real, mu_y_real)

        self._greens = compute_greens_weights(
            greens_model, self.grid.x_ddag, self.grid.y_ddag,
            self.grid.dtau, tol=series_tol,
        )

        self._policies = self._build_policy_grid()
        self._fft_g_pis = self._precompute_policy_kernels()

    @staticmethod
    def _make_real_world_model(
        model: MertonJumpDiffusion2D,
        mu_x_real: float | None,
        mu_y_real: float | None,
    ) -> MertonJumpDiffusion2D:
        import dataclasses

        mu_x = mu_x_real if mu_x_real is not None else model.r
        mu_y = mu_y_real if mu_y_real is not None else model.r

        class _RealWorldModel:
            def __init__(self, base):
                self._base = base

            def __getattr__(self, name):
                return getattr(self._base, name)

            @property
            def beta_tilde(self):
                return np.array([
                    mu_x - 0.5 * self._base.sigma_x**2
                    - self._base.lam * self._base.kappa_x,
                    mu_y - 0.5 * self._base.sigma_y**2
                    - self._base.lam * self._base.kappa_y,
                ])

        return _RealWorldModel(model)

    @staticmethod
    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p

    def _build_policy_grid(self) -> np.ndarray:
        n = self.n_pi
        if self.allow_short:
            pi_vals = np.linspace(-0.5, 1.5, n)
        else:
            pi_vals = np.linspace(0.0, 1.0, n)

        policies = []
        for px in pi_vals:
            for py in pi_vals:
                total = px + py
                if self.allow_short:
                    if total <= 2.0 + 1e-10:
                        policies.append((float(px), float(py)))
                else:
                    if total <= 1.0 + 1e-10:
                        policies.append((float(px), float(py)))
        return np.array(policies, dtype=float)

    def _precompute_policy_kernels(self) -> np.ndarray:
        """Return stacked FFT kernels: shape (n_policies, fft_nx, fft_ny//2+1)."""
        grid = self.grid
        gamma = self.gamma
        dtau = grid.dtau
        rf_factor = np.exp(self.model.r * dtau)
        rn_correction = np.exp(self.model.r * dtau)

        Dx, Dy = np.meshgrid(grid.x_ddag, grid.y_ddag, indexing='ij')
        exp_neg_Dx = np.exp(-Dx)
        exp_neg_Dy = np.exp(-Dy)

        n_pol = len(self._policies)
        # Build all g_pi kernels at once, then batch-FFT
        # g_pis shape: (n_pol, n_disp_x, n_disp_y)
        pi_x_arr = self._policies[:, 0, np.newaxis, np.newaxis]  # (n, 1, 1)
        pi_y_arr = self._policies[:, 1, np.newaxis, np.newaxis]
        pi_c_arr = 1.0 - pi_x_arr - pi_y_arr

        R_p = pi_x_arr * exp_neg_Dx + pi_y_arr * exp_neg_Dy + pi_c_arr * rf_factor
        R_p_pos = np.maximum(R_p, 1e-30)
        g_pis = rn_correction * self._greens * (R_p_pos ** gamma)  # (n_pol, n_disp_x, n_disp_y)

        # Zero-pad into FFT grid
        g_padded = np.zeros((n_pol, self._fft_nx, self._fft_ny), dtype=np.float64)
        g_padded[:, :self._n_disp_x, :self._n_disp_y] = g_pis
        # Batch rfft2 over leading axis (scipy/numpy apply to last two dims)
        return _fft.rfft2(g_padded)  # (n_pol, fft_nx, fft_ny//2+1)

    def _trapezoidal_weights_2d(self, nx: int, ny: int) -> np.ndarray:
        w = np.ones((nx, ny))
        w[0, :] *= 0.5
        w[-1, :] *= 0.5
        w[:, 0] *= 0.5
        w[:, -1] *= 0.5
        return w

    def solve(
        self,
        terminal_utility_func: Callable,
        return_grid_values: bool = False,
        return_policy: bool = False,
        store_policy_times: list | None = None,
    ) -> float | dict:
        grid = self.grid
        model = self.model
        N, J, M = self.N, self.J, self.M

        X_dag, Y_dag = np.meshgrid(grid.x_dag, grid.y_dag, indexing='ij')
        terminal_dag = terminal_utility_func(X_dag, Y_dag)
        V_dag = terminal_dag.copy()

        X_in, Y_in = np.meshgrid(grid.x_in, grid.y_in, indexing='ij')

        trap_w = self._trapezoidal_weights_2d(self._n_dag_x, self._n_dag_y)

        ix_start = N // 2 + 1
        ix_end = ix_start + (N - 1)
        jy_start = J // 2 + 1
        jy_end = jy_start + (J - 1)

        out_x_start = 2 * N
        out_y_start = 2 * J
        n_in_x = N - 1
        n_in_y = J - 1

        stored_policies = {}
        V_interior = None

        _row_idx = np.arange(n_in_x)[:, np.newaxis]
        _col_idx = np.arange(n_in_y)[np.newaxis, :]

        for m in range(M):
            signal = trap_w * V_dag
            sig_padded = np.zeros((self._fft_nx, self._fft_ny))
            sig_padded[:self._n_dag_x, :self._n_dag_y] = signal
            fft_sig = _fft.rfft2(sig_padded)

            # Vectorised: convolve all n_pol kernels in one batched irfft2
            # self._fft_g_pis shape: (n_pol, fft_nx, fft_ny//2+1)
            all_conv = _fft.irfft2(
                self._fft_g_pis * fft_sig,   # broadcast (n_pol,H,W) * (H,W)
                s=(self._fft_nx, self._fft_ny),
            )  # (n_pol, fft_nx, fft_ny)

            # Extract interior for every policy at once → (n_pol, n_in_x, n_in_y)
            all_V = all_conv[
                :,
                out_x_start:out_x_start + n_in_x,
                out_y_start:out_y_start + n_in_y,
            ]

            best_pi_idx = np.argmax(all_V, axis=0)          # (n_in_x, n_in_y)
            V_interior  = all_V[best_pi_idx, _row_idx, _col_idx]

            should_store = (
                (return_policy and m == M - 1)
                or (store_policy_times is not None and m in store_policy_times)
            )
            if should_store:
                stored_policies[m] = (
                    self._policies[best_pi_idx, 0].copy(),
                    self._policies[best_pi_idx, 1].copy(),
                )

            V_dag[ix_start:ix_end, jy_start:jy_end] = V_interior

            V_dag[:ix_start, :] = terminal_dag[:ix_start, :]
            V_dag[ix_end:, :] = terminal_dag[ix_end:, :]
            V_dag[:, :jy_start] = terminal_dag[:, :jy_start]
            V_dag[:, jy_end:] = terminal_dag[:, jy_end:]

        x0_log = np.log(self.X0)
        y0_log = np.log(self.Y0)
        value = self._interpolate(grid.x_in, grid.y_in, V_interior, x0_log, y0_log)

        if not (return_grid_values or return_policy or store_policy_times):
            return value

        result = {'value': value}

        if return_grid_values:
            result['V'] = V_interior.copy()
            result['x'] = grid.x_in.copy()
            result['y'] = grid.y_in.copy()

        if return_policy:
            if M - 1 in stored_policies:
                result['pi_x'] = stored_policies[M - 1][0]
                result['pi_y'] = stored_policies[M - 1][1]
            result['x_in'] = grid.x_in.copy()
            result['y_in'] = grid.y_in.copy()

        if store_policy_times is not None:
            result['policy_snapshots'] = stored_policies
            result['x_in'] = grid.x_in.copy()
            result['y_in'] = grid.y_in.copy()

        return result

    @staticmethod
    def _interpolate(
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        values: np.ndarray,
        x0: float,
        y0: float,
    ) -> float:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (x_grid, y_grid), values,
            method='linear', bounds_error=False, fill_value=None,
        )
        return float(interp([x0, y0]).item())

    @property
    def policies(self) -> np.ndarray:
        return self._policies.copy()

    @property
    def n_policies(self) -> int:
        return len(self._policies)
