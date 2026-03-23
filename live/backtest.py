from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.allocator import TwoAssetAllocator, crra_basket_utility
from live.calibrator import MertonCalibrator, CalibrationResult


_BACKTEST_GRID = dict(N=64, J=64, M=10, n_pi=11)


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    portfolio: np.ndarray
    bench_5050: np.ndarray
    bench_x: np.ndarray
    bench_cash: np.ndarray
    pi_x_series: np.ndarray
    pi_y_series: np.ndarray
    recal_dates: list
    gamma: float
    ticker_x: str
    ticker_y: str

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "portfolio": self.portfolio,
            "bench_5050": self.bench_5050,
            f"all_{self.ticker_x}": self.bench_x,
            "cash": self.bench_cash,
            "pi_x": self.pi_x_series,
            "pi_y": self.pi_y_series,
        }, index=self.dates)

    def metrics(self) -> dict:
        df = self.to_dataframe()
        n = len(df)
        if n < 2:
            return {}

        log_ret = np.log(df[["portfolio", "bench_5050", f"all_{self.ticker_x}", "cash"]]).diff().dropna()

        def _ann(col):
            mu = float(log_ret[col].mean() * 252)
            vol = float(log_ret[col].std() * np.sqrt(252))
            sharpe = mu / vol if vol > 1e-9 else 0.0
            cum = float(df[col].iloc[-1])
            dd = float(_max_drawdown(df[col].values))
            return dict(ann_return=mu, ann_vol=vol, sharpe=sharpe, total_return=cum - 1.0, max_drawdown=dd)

        return {
            "portfolio": _ann("portfolio"),
            "bench_5050": _ann("bench_5050"),
            f"all_{self.ticker_x}": _ann(f"all_{self.ticker_x}"),
            "cash": _ann("cash"),
        }


def _max_drawdown(wealth: np.ndarray) -> float:
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / np.maximum(peak, 1e-12)
    return float(dd.min())


def _solve_allocation(
    cal: CalibrationResult,
    px: float,
    py: float,
    gamma: float,
    alpha: float,
    horizon_years: float,
    grid_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    terminal_util = crra_basket_utility(gamma, alpha=alpha)
    allocator = TwoAssetAllocator(
        cal.model, px, py,
        gamma=gamma,
        mu_x_real=cal.mu_x_real,
        mu_y_real=cal.mu_y_real,
        allow_short=False,
        domain_half_width_x=0.8,
        domain_half_width_y=0.8,
        **grid_kwargs,
    )
    res = allocator.solve(terminal_util, return_grid_values=True, return_policy=True)
    return res["x"], res["y"], res["pi_x"], res["pi_y"]


def _build_interpolators(
    x_grid: np.ndarray, y_grid: np.ndarray,
    pi_x_grid: np.ndarray, pi_y_grid: np.ndarray,
):
    from scipy.interpolate import RegularGridInterpolator
    kw = dict(method="linear", bounds_error=False, fill_value=None)
    return (
        RegularGridInterpolator((x_grid, y_grid), pi_x_grid, **kw),
        RegularGridInterpolator((x_grid, y_grid), pi_y_grid, **kw),
    )


def _interpolate_policy(
    log_x: float, log_y: float,
    ix, iy,
) -> tuple[float, float]:
    pt = [[log_x, log_y]]
    pi_x = float(np.clip(ix(pt).item(), 0, 1))
    pi_y = float(np.clip(iy(pt).item(), 0, 1))
    total = pi_x + pi_y
    if total > 1.0:
        pi_x /= total
        pi_y /= total
    return pi_x, pi_y


def run_backtest(
    log_returns: pd.DataFrame,
    risk_free_rates: Optional[pd.Series] = None,
    gamma: float = -1.0,
    alpha: float = 0.5,
    horizon_years: float = 1.0,
    cal_window_days: int = 504,
    rebal_freq_days: int = 21,
    grid_kwargs: Optional[dict] = None,
    ticker_x: str = "X",
    ticker_y: str = "Y",
    progress_cb=None,
    surrogate=None,
) -> BacktestResult:
    if grid_kwargs is None:
        grid_kwargs = _BACKTEST_GRID

    calibrator = MertonCalibrator(horizon_years=horizon_years)
    rx = log_returns["rx"].values
    ry = log_returns["ry"].values
    dates = log_returns.index
    n = len(rx)

    if risk_free_rates is None:
        rfr_arr = np.full(n, 0.05)
    else:
        rfr_arr = risk_free_rates.reindex(dates).ffill().fillna(0.05).values

    portfolio = np.ones(n)
    bench_5050 = np.ones(n)
    bench_x = np.ones(n)
    bench_cash = np.ones(n)
    pi_x_series = np.zeros(n)
    pi_y_series = np.zeros(n)
    recal_dates = []

    pi_x = pi_y = 0.0
    ix = iy = None
    last_px = last_py = 1.0

    rebal_steps = [i for i in range(cal_window_days, n, rebal_freq_days)]
    total_steps = len(rebal_steps)

    for step_idx, i in enumerate(rebal_steps):
        window = log_returns.iloc[i - cal_window_days: i]
        rfr = float(rfr_arr[i])

        try:
            cal = calibrator.fit(window, risk_free_rate=rfr)
        except Exception:
            pi_x_series[i] = pi_x
            pi_y_series[i] = pi_y
            continue

        log_px = float(np.sum(rx[i - cal_window_days: i]))
        log_py = float(np.sum(ry[i - cal_window_days: i]))
        px = np.exp(log_px)
        py = np.exp(log_py)

        try:
            if surrogate is not None:
                pi_x, pi_y = surrogate.predict(
                    cal.model.sigma_x, cal.model.sigma_y, cal.model.rho,
                    cal.model.lam, cal.mu_x_real, cal.mu_y_real, cal.model.r,
                    dx=0.0, dy=0.0,
                )
            else:
                x_grid, y_grid, pi_x_grid, pi_y_grid = _solve_allocation(
                    cal, px, py, gamma, alpha, horizon_years, grid_kwargs,
                )
                ix, iy = _build_interpolators(x_grid, y_grid, pi_x_grid, pi_y_grid)
                pi_x, pi_y = _interpolate_policy(log_px, log_py, ix, iy)
        except Exception:
            pass

        recal_dates.append(dates[i])
        last_px = px
        last_py = py

        pi_x_series[i] = pi_x
        pi_y_series[i] = pi_y

        if progress_cb is not None:
            progress_cb(step_idx + 1, total_steps)

    rebal_set = set(rebal_steps)
    for i in range(1, n):
        if i not in rebal_set:
            pi_x_series[i] = pi_x_series[i - 1]
            pi_y_series[i] = pi_y_series[i - 1]

    pi_cash_arr = np.clip(1.0 - pi_x_series - pi_y_series, 0.0, 1.0)

    for i in range(1, n):
        daily_rfr = rfr_arr[i] / 252.0
        port_ret = pi_x_series[i - 1] * rx[i] + pi_y_series[i - 1] * ry[i] + pi_cash_arr[i - 1] * daily_rfr
        portfolio[i] = portfolio[i - 1] * np.exp(port_ret)
        bench_5050[i] = bench_5050[i - 1] * np.exp(0.5 * rx[i] + 0.5 * ry[i])
        bench_x[i] = bench_x[i - 1] * np.exp(rx[i])
        bench_cash[i] = bench_cash[i - 1] * np.exp(daily_rfr)

    return BacktestResult(
        dates=dates,
        portfolio=portfolio,
        bench_5050=bench_5050,
        bench_x=bench_x,
        bench_cash=bench_cash,
        pi_x_series=pi_x_series,
        pi_y_series=pi_y_series,
        recal_dates=recal_dates,
        gamma=gamma,
        ticker_x=ticker_x,
        ticker_y=ticker_y,
    )


def bootstrap_ci(
    bt: BacktestResult,
    n_boot: int = 500,
    block_len: int = 21,
    ci_level: float = 0.95,
) -> dict:
    df = bt.to_dataframe()
    cols = ["portfolio", "bench_5050", f"all_{bt.ticker_x}", "cash"]

    log_rets = {}
    for c in cols:
        if c in df.columns:
            arr = df[c].values
            log_rets[c] = np.log(np.maximum(arr[1:] / arr[:-1], 1e-30))

    n = len(next(iter(log_rets.values())))
    alpha = (1.0 - ci_level) / 2.0
    rng = np.random.default_rng(42)
    max_start = max(n - block_len, 1)
    n_blocks = int(np.ceil(n / block_len))

    boot_stats: dict = {c: {"ann_return": [], "sharpe": [], "max_drawdown": []} for c in log_rets}

    for _ in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([
            np.arange(s, min(s + block_len, n)) for s in starts
        ])[:n]

        for c, lr in log_rets.items():
            boot_lr = lr[idx]
            mu = float(boot_lr.mean() * 252)
            vol = float(boot_lr.std() * np.sqrt(252))
            sharpe = mu / vol if vol > 1e-9 else 0.0
            wealth = np.concatenate([[1.0], np.exp(np.cumsum(boot_lr))])
            dd = float(_max_drawdown(wealth))
            boot_stats[c]["ann_return"].append(mu)
            boot_stats[c]["sharpe"].append(sharpe)
            boot_stats[c]["max_drawdown"].append(dd)

    result = {}
    for c, stats in boot_stats.items():
        result[c] = {}
        for metric, vals in stats.items():
            arr = np.array(vals)
            result[c][metric] = {
                "mean": float(arr.mean()),
                "ci_low": float(np.quantile(arr, alpha)),
                "ci_high": float(np.quantile(arr, 1.0 - alpha)),
            }
    return result
