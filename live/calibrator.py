from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import MertonJumpDiffusion2D


_LAMBDA_GRID = np.linspace(0.0, 1.0, 41)


def _optimal_shrinkage(
    rx_train: np.ndarray, ry_train: np.ndarray,
    rx_val: np.ndarray, ry_val: np.ndarray,
    mu_x_prior: float, mu_y_prior: float,
    var_x: float, var_y: float,
) -> float:
    if len(rx_val) < 20 or var_x < 1e-12 or var_y < 1e-12:
        return 0.0

    mu_x_tr = float(np.mean(rx_train) * 252)
    mu_y_tr = float(np.mean(ry_train) * 252)

    lam = _LAMBDA_GRID
    mu_x_c = ((1.0 - lam) * mu_x_tr + lam * mu_x_prior)[:, None] / 252
    mu_y_c = ((1.0 - lam) * mu_y_tr + lam * mu_y_prior)[:, None] / 252

    resid_x = rx_val[None, :] - mu_x_c
    resid_y = ry_val[None, :] - mu_y_c

    ll = -(resid_x ** 2 / var_x + resid_y ** 2 / var_y).sum(axis=1)
    return float(lam[int(np.argmax(ll))])


@dataclass
class CalibrationResult:
    model: MertonJumpDiffusion2D
    mu_x_real: float
    mu_y_real: float
    mu_x_raw: float
    mu_y_raw: float
    beta_x: float
    beta_y: float
    mu_x_prior: float
    mu_y_prior: float
    shrinkage: float
    n_total: int
    n_jump: int
    lambda_raw: float

    def __repr__(self) -> str:
        m = self.model
        return (
            f"CalibrationResult(\n"
            f"  sigma_x={m.sigma_x:.4f}  sigma_y={m.sigma_y:.4f}  rho={m.rho:.3f}\n"
            f"  lam={m.lam:.3f}  mu_tilde_x={m.mu_tilde_x:.4f}  mu_tilde_y={m.mu_tilde_y:.4f}\n"
            f"  sigma_tilde_x={m.sigma_tilde_x:.4f}  sigma_tilde_y={m.sigma_tilde_y:.4f}  rho_hat={m.rho_hat:.3f}\n"
            f"  r={m.r:.4f}  T={m.T:.2f}\n"
            f"  mu_x_real={self.mu_x_real:.4f} (raw={self.mu_x_raw:.4f} prior={self.mu_x_prior:.4f} β={self.beta_x:.2f})\n"
            f"  mu_y_real={self.mu_y_real:.4f} (raw={self.mu_y_raw:.4f} prior={self.mu_y_prior:.4f} β={self.beta_y:.2f})\n"
            f"  auto λ={self.shrinkage:.2f}\n"
            f"  jump_days={self.n_jump}/{self.n_total} ({100*self.n_jump/self.n_total:.1f}%)\n"
            f")"
        )


class MertonCalibrator:

    def __init__(
        self,
        jump_threshold: float = 2.5,
        max_iter: int = 5,
        horizon_years: float = 1.0,
        market_premium: float = 0.055,
        val_frac: float = 0.20,
    ):
        self.jump_threshold = jump_threshold
        self.max_iter = max_iter
        self.horizon_years = horizon_years
        self.market_premium = market_premium
        self.val_frac = val_frac

    def fit(
        self,
        log_returns: pd.DataFrame,
        risk_free_rate: float = 0.05,
    ) -> CalibrationResult:
        rx = log_returns["rx"].values
        ry = log_returns["ry"].values
        n_total = len(rx)
        if n_total < 30:
            raise ValueError(
                f"Insufficient data: only {n_total} rows returned. "
                "Check tickers and data feed — yfinance may have returned empty data."
            )

        mu_x_raw = float(np.mean(rx) * 252)
        mu_y_raw = float(np.mean(ry) * 252)

        r_m = (rx + ry) * 0.5
        cov = np.cov(np.stack([rx, ry, r_m], axis=0))
        var_m = cov[2, 2]
        if var_m > 1e-12:
            beta_x = float(cov[0, 2] / var_m)
            beta_y = float(cov[1, 2] / var_m)
        else:
            beta_x = beta_y = 1.0

        r = float(np.clip(risk_free_rate, 0.001, 0.20))
        mu_x_prior = r + beta_x * self.market_premium
        mu_y_prior = r + beta_y * self.market_premium

        n_train = int(n_total * (1.0 - self.val_frac))
        cov_train = np.cov(np.stack([rx[:n_train], ry[:n_train]], axis=0))
        shrinkage = _optimal_shrinkage(
            rx[:n_train], ry[:n_train],
            rx[n_train:], ry[n_train:],
            mu_x_prior, mu_y_prior,
            float(cov_train[0, 0]), float(cov_train[1, 1]),
        )

        mu_x_real = (1.0 - shrinkage) * mu_x_raw + shrinkage * mu_x_prior
        mu_y_real = (1.0 - shrinkage) * mu_y_raw + shrinkage * mu_y_prior

        is_jump = np.zeros(n_total, dtype=bool)
        for _ in range(self.max_iter):
            non_jump = ~is_jump
            if non_jump.sum() < 10:
                non_jump = np.ones(n_total, dtype=bool)
            sx = max(float(np.std(rx[non_jump])), 1e-4)
            sy = max(float(np.std(ry[non_jump])), 1e-4)
            is_jump = (np.abs(rx) > self.jump_threshold * sx) | (np.abs(ry) > self.jump_threshold * sy)

        non_jump = ~is_jump
        n_nj = max(int(non_jump.sum()), 1)
        rx_nj = rx[non_jump]
        ry_nj = ry[non_jump]

        sigma_x = float(np.std(rx_nj) * np.sqrt(252))
        sigma_y = float(np.std(ry_nj) * np.sqrt(252))
        rho = float(np.corrcoef(rx_nj, ry_nj)[0, 1]) if n_nj > 2 else 0.0

        n_jump = int(is_jump.sum())
        lambda_raw = float(n_jump) / n_total * 252.0

        if n_jump >= 4:
            jx = rx[is_jump]
            jy = ry[is_jump]
            mu_tilde_x = float(np.mean(jx))
            mu_tilde_y = float(np.mean(jy))
            sigma_tilde_x = max(float(np.std(jx)), 0.01)
            sigma_tilde_y = max(float(np.std(jy)), 0.01)
            rho_hat = float(np.corrcoef(jx, jy)[0, 1]) if n_jump > 4 else 0.0
        else:
            mu_tilde_x = mu_tilde_y = -0.02
            sigma_tilde_x = sigma_tilde_y = 0.05
            rho_hat = 0.0

        model = MertonJumpDiffusion2D(
            sigma_x=float(np.clip(sigma_x, 0.01, 2.0)),
            sigma_y=float(np.clip(sigma_y, 0.01, 2.0)),
            rho=float(np.clip(rho, -0.98, 0.98)),
            lam=float(np.clip(lambda_raw, 0.0, 50.0)),
            mu_tilde_x=float(mu_tilde_x),
            mu_tilde_y=float(mu_tilde_y),
            sigma_tilde_x=float(np.clip(sigma_tilde_x, 0.005, 2.0)),
            sigma_tilde_y=float(np.clip(sigma_tilde_y, 0.005, 2.0)),
            rho_hat=float(np.clip(rho_hat, -0.98, 0.98)),
            r=float(r),
            T=float(self.horizon_years),
            K=1.0,
        )

        return CalibrationResult(
            model=model,
            mu_x_real=float(mu_x_real),
            mu_y_real=float(mu_y_real),
            mu_x_raw=float(mu_x_raw),
            mu_y_raw=float(mu_y_raw),
            beta_x=float(beta_x),
            beta_y=float(beta_y),
            mu_x_prior=float(mu_x_prior),
            mu_y_prior=float(mu_y_prior),
            shrinkage=float(shrinkage),
            n_total=n_total,
            n_jump=n_jump,
            lambda_raw=lambda_raw,
        )

    def rolling_fit(
        self,
        log_returns: pd.DataFrame,
        risk_free_rate: float = 0.05,
        window_days: int = 252,
    ) -> list[CalibrationResult]:
        results = []
        for end in range(window_days, len(log_returns) + 1):
            window = log_returns.iloc[end - window_days:end]
            results.append(self.fit(window, risk_free_rate=risk_free_rate))
        return results
