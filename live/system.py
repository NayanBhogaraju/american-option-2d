from __future__ import annotations

import time
import os
import sys as _sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.allocator import TwoAssetAllocator, crra_basket_utility
from live.data_feed import DataFeed
from live.calibrator import MertonCalibrator, CalibrationResult
from live.policy_net import PolicyNet, train_policy_net


DEFAULT_GRID = dict(N=128, J=128, M=20, n_pi=11)


@dataclass
class AllocationSystem:

    ticker_x: str = "SPY"
    ticker_y: str = "QQQ"
    gamma: float = -1.0
    alpha: float = 0.5
    horizon_years: float = 1.0
    grid_kwargs: dict = field(default_factory=lambda: dict(**DEFAULT_GRID))
    net_epochs: int = 300

    _feed: DataFeed = field(init=False, repr=False)
    _calibrator: MertonCalibrator = field(init=False, repr=False)
    _cal_result: Optional[CalibrationResult] = field(default=None, init=False, repr=False)
    _net: Optional[PolicyNet] = field(default=None, init=False, repr=False)
    _solver_result: Optional[dict] = field(default=None, init=False, repr=False)
    _last_pipeline_time: float = field(default=0.0, init=False, repr=False)
    _pipeline_duration_s: float = field(default=0.0, init=False, repr=False)
    _calibration_prices: tuple = field(default=(None, None), init=False, repr=False)
    _grid_interp_x: object = field(default=None, init=False, repr=False)
    _grid_interp_y: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._feed = DataFeed(ticker_x=self.ticker_x, ticker_y=self.ticker_y,
                               lookback_years=2)
        self._calibrator = MertonCalibrator(horizon_years=self.horizon_years)

    def run_full_pipeline(self, verbose: bool = True) -> None:
        for _step, _msg in self.run_pipeline_steps(verbose=verbose):
            pass

    def run_pipeline_steps(self, verbose: bool = True):
        t0 = time.time()

        if verbose:
            print(f"[System] Fetching data for {self.ticker_x} / {self.ticker_y} ...")
        self._feed.history(refresh=True)
        r = self._feed.risk_free_rate(refresh=True)
        yield 1, f"Data fetched ({len(self._feed.log_returns())} trading days)"

        if verbose:
            print("[System] Calibrating Merton model ...")
        rets = self._feed.log_returns()
        self._cal_result = self._calibrator.fit(rets, risk_free_rate=r)
        if verbose:
            print(self._cal_result)
        yield 2, (
            f"Model calibrated  σ_x={self._cal_result.model.sigma_x:.3f}  "
            f"σ_y={self._cal_result.model.sigma_y:.3f}  "
            f"λ={self._cal_result.model.lam:.2f}  "
            f"jump days={self._cal_result.n_jump}/{self._cal_result.n_total}"
        )

        if verbose:
            print("[System] Solving Bellman equation ...")
        self._run_solver(verbose=verbose)
        yield 3, f"Bellman solved  V(X0,Y0)={self._solver_result['value']:.4f}"

        if verbose:
            print("[System] Training policy network ...")
        self._train_net(verbose=verbose)
        yield 4, "Policy network trained"

        self._last_pipeline_time = time.time()
        self._pipeline_duration_s = time.time() - t0
        if verbose:
            print(f"[System] Pipeline complete in {self._pipeline_duration_s:.1f}s")

    def allocate(
        self,
        X_price: Optional[float] = None,
        Y_price: Optional[float] = None,
    ) -> tuple[float, float]:
        if X_price is None or Y_price is None:
            X_price, Y_price = self._feed.current_prices()

        log_x = np.log(float(X_price))
        log_y = np.log(float(Y_price))

        if self._net is not None:
            return self._net.predict(log_x, log_y)

        if self._solver_result is not None:
            return self._grid_interpolate(log_x, log_y)

        raise RuntimeError("Pipeline has not been run yet. Call run_full_pipeline().")

    def status(self) -> dict:
        try:
            px, py = self._feed.current_prices()
        except Exception:
            px = py = float("nan")

        pi_x = pi_y = float("nan")
        try:
            pi_x, pi_y = self.allocate(px, py)
        except Exception:
            pass

        cal = self._cal_result
        return {
            "ticker_x": self.ticker_x,
            "ticker_y": self.ticker_y,
            "X_price": px,
            "Y_price": py,
            "pi_x": pi_x,
            "pi_y": pi_y,
            "pi_cash": max(0.0, 1.0 - pi_x - pi_y) if np.isfinite(pi_x) else float("nan"),
            "gamma": self.gamma,
            "horizon_years": self.horizon_years,
            "sigma_x": cal.model.sigma_x if cal else None,
            "sigma_y": cal.model.sigma_y if cal else None,
            "rho": cal.model.rho if cal else None,
            "lam": cal.model.lam if cal else None,
            "mu_x_real": cal.mu_x_real if cal else None,
            "mu_y_real": cal.mu_y_real if cal else None,
            "mu_x_raw": cal.mu_x_raw if cal else None,
            "mu_y_raw": cal.mu_y_raw if cal else None,
            "mu_x_prior": cal.mu_x_prior if cal else None,
            "mu_y_prior": cal.mu_y_prior if cal else None,
            "beta_x": cal.beta_x if cal else None,
            "beta_y": cal.beta_y if cal else None,
            "shrinkage": cal.shrinkage if cal else None,
            "n_jump_days": cal.n_jump if cal else None,
            "n_total_days": cal.n_total if cal else None,
            "equity_premium_x": (cal.mu_x_real - cal.model.r) if cal else None,
            "equity_premium_y": (cal.mu_y_real - cal.model.r) if cal else None,
            "hurdle_x": (cal.model.r + 1.5 * cal.model.sigma_x ** 2) if cal else None,
            "hurdle_y": (cal.model.r + 1.5 * cal.model.sigma_y ** 2) if cal else None,
            "r": cal.model.r if cal else None,
            "expected_return_optimal": self._expected_return(pi_x, pi_y, cal) if cal and np.isfinite(pi_x) else None,
            "expected_return_50_50":   self._expected_return(0.5, 0.5, cal) if cal else None,
            "expected_return_all_x":   self._expected_return(1.0, 0.0, cal) if cal else None,
            "expected_return_cash":    cal.model.r if cal else None,
            "X_entry": self._calibration_prices[0],
            "Y_entry": self._calibration_prices[1],
            "pipeline_duration_s": self._pipeline_duration_s,
            "last_pipeline_time": self._last_pipeline_time,
            "net_ready": self._net is not None,
            "solver_ready": self._solver_result is not None,
        }

    def policy_surface(self) -> Optional[dict]:
        if self._solver_result is None:
            return None
        res = self._solver_result
        return {
            "x": res["x"],
            "y": res["y"],
            "pi_x": res["pi_x"],
            "pi_y": res["pi_y"],
            "V": res.get("V"),
        }

    @staticmethod
    def _expected_return(pi_x: float, pi_y: float, cal) -> float:
        pi_cash = max(0.0, 1.0 - pi_x - pi_y)
        return float(pi_x * cal.mu_x_real + pi_y * cal.mu_y_real + pi_cash * cal.model.r)

    def _run_solver(self, verbose: bool = True) -> None:
        cal = self._cal_result
        if cal is None:
            raise RuntimeError("Must calibrate before solving.")

        px, py = self._feed.current_prices()
        self._calibration_prices = (px, py)

        terminal_util = crra_basket_utility(self.gamma, alpha=self.alpha)
        allocator = TwoAssetAllocator(
            cal.model, px, py,
            gamma=self.gamma,
            mu_x_real=cal.mu_x_real,
            mu_y_real=cal.mu_y_real,
            allow_short=False,
            domain_half_width_x=0.8,
            domain_half_width_y=0.8,
            **self.grid_kwargs,
        )
        res = allocator.solve(
            terminal_util,
            return_grid_values=True,
            return_policy=True,
        )
        self._solver_result = res
        from scipy.interpolate import RegularGridInterpolator
        kw = dict(method="linear", bounds_error=False, fill_value=None)
        self._grid_interp_x = RegularGridInterpolator((res["x"], res["y"]), res["pi_x"], **kw)
        self._grid_interp_y = RegularGridInterpolator((res["x"], res["y"]), res["pi_y"], **kw)
        if verbose:
            print(f"  Solver value at (X0, Y0): {res['value']:.6f}")

    def _train_net(self, verbose: bool = True) -> None:
        if self._solver_result is None:
            raise RuntimeError("Must solve before training net.")

        res = self._solver_result
        net = PolicyNet(hidden=64)
        net = train_policy_net(
            net,
            res["x"], res["y"],
            res["pi_x"], res["pi_y"],
            epochs=self.net_epochs,
            verbose=verbose,
        )
        self._net = net

    def _grid_interpolate(self, log_x: float, log_y: float) -> tuple[float, float]:
        pt = [[log_x, log_y]]
        pi_x = float(np.clip(self._grid_interp_x(pt).item(), 0, 1))
        pi_y = float(np.clip(self._grid_interp_y(pt).item(), 0, 1))
        total = pi_x + pi_y
        if total > 1.0:
            pi_x /= total
            pi_y /= total
        return pi_x, pi_y
