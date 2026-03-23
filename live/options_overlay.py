from __future__ import annotations

import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm


def _bs_put(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bs_call(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_delta_put(S: float, K: float, r: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) - 1.0)


def indifference_put_price(
    model,
    px: float,
    py: float,
    gamma: float,
    alpha: float,
    mu_x_real: float,
    mu_y_real: float,
    moneyness: float,
    V_no_put: float,
) -> float:
    """
    Utility-indifference price for a protective put on the X asset.

    The put floors the X component of the basket at the normalized strike k = moneyness:
      U_put(x, y) = (alpha * max(e^x, k) + (1-alpha) * e^y)^gamma / gamma

    Indifference price (as fraction of W0):
      p* / W0 = 1 - (V_no_put / V_with_put)^(1/gamma)

    Returns p* as a fraction of initial portfolio value.
    """
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from core.allocator import TwoAssetAllocator

    k = float(moneyness)

    def _put_terminal(x, y):
        basket = alpha * np.maximum(np.exp(x), k) + (1.0 - alpha) * np.exp(y)
        return np.power(np.maximum(basket, 1e-30), gamma) / gamma

    try:
        allocator = TwoAssetAllocator(
            model, px, py,
            gamma=gamma,
            mu_x_real=mu_x_real,
            mu_y_real=mu_y_real,
            allow_short=False,
            domain_half_width_x=0.8,
            domain_half_width_y=0.8,
            N=64, J=64, M=5, n_pi=11,
        )
        V_with_put = float(allocator.solve(_put_terminal))
    except Exception:
        return 0.0

    if V_with_put == 0.0 or V_no_put == 0.0:
        return 0.0

    ratio = V_no_put / V_with_put
    if ratio <= 0.0:
        return 0.0

    p_frac = 1.0 - ratio ** (1.0 / gamma)
    return max(0.0, float(p_frac))


@dataclass
class HedgeQuote:
    ticker: str
    S: float
    K: float
    T: float
    sigma: float
    r: float
    position_value: float

    put_price_per_share: float
    put_price_pct: float
    delta: float

    hedged_position_value: float
    hedged_downside_1sd: float
    unhedged_downside_1sd: float
    breakeven_return: float

    collar_call_price: float
    collar_net_cost_pct: float
    collar_upside_cap_pct: float


def price_protective_put(
    ticker: str,
    S: float,
    sigma: float,
    r: float,
    position_value: float,
    T: float = 0.25,
    moneyness: float = 0.95,
) -> HedgeQuote:
    K = S * moneyness
    put_price = _bs_put(S, K, r, T, sigma)
    put_pct = put_price / S

    hedged_val = position_value - position_value * put_pct
    delta = _bs_delta_put(S, K, r, T, sigma)
    shock = sigma * np.sqrt(T)
    unhedged_down = position_value * (np.exp(-shock) - 1.0)
    hedged_down = position_value * (np.exp(-shock) - 1.0) + position_value * max(K - S * np.exp(-shock), 0.0) / S

    call_K = S * (1.0 + moneyness * 0.1 + 0.05)
    call_price = _bs_call(S, call_K, r, T, sigma)
    collar_net = (put_price - call_price) / S
    collar_cap = (call_K / S) - 1.0

    breakeven = -put_pct

    return HedgeQuote(
        ticker=ticker,
        S=S,
        K=K,
        T=T,
        sigma=sigma,
        r=r,
        position_value=position_value,
        put_price_per_share=put_price,
        put_price_pct=put_pct,
        delta=delta,
        hedged_position_value=hedged_val,
        hedged_downside_1sd=hedged_down,
        unhedged_downside_1sd=unhedged_down,
        breakeven_return=breakeven,
        collar_call_price=call_price,
        collar_net_cost_pct=collar_net,
        collar_upside_cap_pct=collar_cap,
    )


def options_overlay(
    ticker_x: str,
    ticker_y: str,
    S_x: float,
    S_y: float,
    sigma_x: float,
    sigma_y: float,
    r: float,
    pi_x: float,
    pi_y: float,
    account_balance: float,
    T: float = 0.25,
    moneyness: float = 0.95,
    system=None,
) -> dict:
    pos_x = account_balance * pi_x
    pos_y = account_balance * pi_y

    quotes = {}
    if pos_x > 0:
        quotes[ticker_x] = price_protective_put(ticker_x, S_x, sigma_x, r, pos_x, T, moneyness)
    if pos_y > 0:
        quotes[ticker_y] = price_protective_put(ticker_y, S_y, sigma_y, r, pos_y, T, moneyness)

    total_hedge_cost = sum(q.put_price_pct * q.position_value for q in quotes.values())
    total_equity = pos_x + pos_y
    total_cost_pct = total_hedge_cost / account_balance if account_balance > 0 else 0.0

    indif_price_pct = None
    indif_price_dollars = None
    if system is not None:
        try:
            cal = system._cal_result
            sol = system._solver_result
            if cal is not None and sol is not None:
                indif_price_pct = indifference_put_price(
                    model=cal.model,
                    px=system._calibration_prices[0],
                    py=system._calibration_prices[1],
                    gamma=system.gamma,
                    alpha=system.alpha,
                    mu_x_real=cal.mu_x_real,
                    mu_y_real=cal.mu_y_real,
                    moneyness=moneyness,
                    V_no_put=sol["value"],
                )
                indif_price_dollars = indif_price_pct * account_balance
        except Exception:
            pass

    return {
        "quotes": quotes,
        "total_hedge_cost_dollars": total_hedge_cost,
        "total_hedge_cost_pct": total_cost_pct,
        "total_equity_value": total_equity,
        "indif_price_pct": indif_price_pct,
        "indif_price_dollars": indif_price_dollars,
    }


def vol_surface_smile(
    S: float, r: float, T: float, atm_vol: float, n_strikes: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    moneyness_arr = np.linspace(0.80, 1.10, n_strikes)
    strikes = S * moneyness_arr
    skew = 0.05 * (1.0 - moneyness_arr)
    vols = np.clip(atm_vol + skew, 0.01, 2.0)
    puts = np.array([_bs_put(S, K, r, T, v) for K, v in zip(strikes, vols)])
    return strikes, vols, puts
