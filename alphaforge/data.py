"""Price data loading.

Real data comes from Yahoo Finance (via yfinance) and is cached on disk as CSV.
Synthetic data exists only to test the engine: it is a random walk, so it has
no edge by construction and must never be used to judge a strategy.
"""
from __future__ import annotations

import re
import time
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


class DataError(RuntimeError):
    """Raised when price data cannot be loaded."""


def synthetic_prices(days: int = 1500, seed: int = 42, start_price: float = 100.0,
                     daily_vol: float = 0.01, daily_drift: float = 0.0) -> pd.Series:
    """Geometric random walk on business days. Reproducible via `seed`."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(daily_drift, daily_vol, size=days)
    prices = start_price * np.exp(np.cumsum(log_returns))
    index = pd.bdate_range("2015-01-01", periods=days)
    return pd.Series(prices, index=index, name="SYNTHETIC")


def _cache_path(cache_dir: Path, ticker: str, start: str, end: str | None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)
    return cache_dir / f"{safe}_{start}_{end or 'latest'}.csv"


def load_prices(ticker: str, start: str = "2012-01-01", end: str | None = None,
                cache_dir: str | Path | None = None) -> pd.Series:
    """Adjusted daily closes (splits and dividends adjusted)."""
    if cache_dir is None:
        # Serverless deployments have an ephemeral, writable /tmp directory.
        # Keep the familiar project-local cache for desktop development.
        cache_dir = (Path(tempfile.gettempdir()) / "alphaforge-data-cache"
                     if os.environ.get("VERCEL") else Path("data_cache"))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, ticker, start, end)

    # A fixed end date never changes, so cache forever. Open-ended: refresh daily.
    if path.exists() and (end is not None or time.time() - path.stat().st_mtime < 86_400):
        series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        series.name = ticker
        return series

    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError("yfinance is not installed. Run: pip install yfinance") from exc

    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         progress=False)
    except Exception as exc:  # network errors, rate limits, bad symbols
        raise DataError(f"Could not download {ticker}: {exc}") from exc

    if df is None or df.empty or "Close" not in df:
        raise DataError(f"No price data returned for '{ticker}'. Check the symbol "
                        f"(NSE stocks end in .NS, indices look like ^NSEI).")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):          # newer yfinance returns MultiIndex columns
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = ticker
    close.to_csv(path, header=True)
    return close
