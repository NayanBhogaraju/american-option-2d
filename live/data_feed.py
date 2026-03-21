import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class DataFeed:
    ticker_x: str = "SPY"
    ticker_y: str = "QQQ"
    lookback_years: int = 5
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def history(self, refresh: bool = False) -> pd.DataFrame:
        key = "history"
        if key in self._cache and not refresh:
            return self._cache[key]

        end = datetime.date.today()
        start = end - datetime.timedelta(days=int(self.lookback_years * 365.25))

        raw = yf.download(
            [self.ticker_x, self.ticker_y],
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"][[self.ticker_x, self.ticker_y]].copy()
        else:
            close = raw[["Close"]].copy()

        close.columns = ["X_close", "Y_close"]
        close.dropna(inplace=True)
        self._cache[key] = close
        return close

    def log_returns(self, refresh: bool = False) -> pd.DataFrame:
        closes = self.history(refresh=refresh)
        rets = np.log(closes / closes.shift(1)).dropna()
        rets.columns = ["rx", "ry"]
        return rets

    def current_prices(self) -> tuple[float, float]:
        px = _latest_price(self.ticker_x)
        py = _latest_price(self.ticker_y)
        return px, py

    def risk_free_rate(self, refresh: bool = False) -> float:
        key = "rfr"
        if key in self._cache and not refresh:
            return self._cache[key]

        try:
            irx = yf.Ticker("^IRX")
            hist = irx.history(period="5d")
            if hist.empty:
                raise ValueError("empty")
            rate = float(hist["Close"].iloc[-1]) / 100.0
        except Exception:
            rate = 0.05

        rate = max(0.001, min(rate, 0.20))
        self._cache[key] = rate
        return rate

    def summary(self) -> dict:
        closes = self.history()
        px, py = self.current_prices()
        r = self.risk_free_rate()
        rets = self.log_returns()
        return {
            "ticker_x": self.ticker_x,
            "ticker_y": self.ticker_y,
            "X_current": px,
            "Y_current": py,
            "X_last_close": float(closes["X_close"].iloc[-1]),
            "Y_last_close": float(closes["Y_close"].iloc[-1]),
            "risk_free_rate": r,
            "n_days": len(closes),
            "date_range": (closes.index[0].date(), closes.index[-1].date()),
            "X_ann_vol": float(rets["rx"].std() * np.sqrt(252)),
            "Y_ann_vol": float(rets["ry"].std() * np.sqrt(252)),
            "XY_corr": float(rets[["rx", "ry"]].corr().iloc[0, 1]),
        }


def _latest_price(ticker: str) -> float:
    t = yf.Ticker(ticker)
    try:
        info = t.fast_info
        price = float(info.last_price)
        if np.isfinite(price) and price > 0:
            return price
    except Exception:
        pass
    hist = t.history(period="2d", auto_adjust=True)
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    raise RuntimeError(f"Could not fetch price for {ticker}")
