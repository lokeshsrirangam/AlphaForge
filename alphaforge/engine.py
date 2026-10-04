"""Vectorised backtest engine for a moving-average crossover trend strategy.

Timeline convention (the most important thing in this file)
-----------------------------------------------------------
    day t    : we see close[t], compute indicators and decide target[t]
    day t+1  : the order executes at close[t+1]   (execution_lag_days = 1)
    day t+2  : the position earns return[t+2] = close[t+2] / close[t+1] - 1

So  held[k] = target[k - 1 - execution_lag_days].
With execution_lag_days = 0 you trade at the same close you used to compute the
signal. That is only achievable if you can compute and send the order before the
close, so it is the optimistic setting.

Vocabulary used throughout
--------------------------
    raw signal : 1 when fast MA > slow MA, else 0     (a *candidate* to be long)
    size       : volatility-targeted position fraction (0..max_leverage)
    target     : signal x size, decided on day t
    held       : position actually held while a day's return is earned
    trade      : one contiguous run of held > 0
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    fast: int = 10
    slow: int = 30
    ma_type: str = "sma"                # "sma" or "ema"
    stop_loss_pct: float = 0.0          # close-based stop, 0 disables
    target_vol: float | None = 0.15     # annualised; None = always fully invested
    vol_lookback: int = 20
    max_leverage: float = 1.0           # 1.0 = long-only, no leverage
    cost_bps: float = 10.0              # per unit of turnover (brokerage + tax + slippage)
    execution_lag_days: int = 1         # 1 = trade next close, 0 = same close (optimistic)
    risk_free: float = 0.0              # annual rate used in Sharpe / Sortino
    periods_per_year: int = 252         # 365 for crypto
    initial_capital: float = 100_000.0

    def validate(self) -> None:
        if self.ma_type not in ("sma", "ema"):
            raise ValueError("ma_type must be 'sma' or 'ema'")
        if not (2 <= self.fast < self.slow):
            raise ValueError("Need 2 <= fast < slow")
        if self.stop_loss_pct < 0:
            raise ValueError("stop_loss_pct cannot be negative")
        if self.target_vol is not None and self.target_vol <= 0:
            raise ValueError("target_vol must be positive (or None to disable)")
        if self.vol_lookback < 5:
            raise ValueError("vol_lookback must be at least 5")
        if not (0 < self.max_leverage <= 5):
            raise ValueError("max_leverage must be in (0, 5]")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        if self.execution_lag_days not in (0, 1, 2):
            raise ValueError("execution_lag_days must be 0, 1 or 2")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

    @property
    def warmup(self) -> int:
        """Index of the first bar that can possibly hold a position."""
        needed = self.slow - 1
        if self.target_vol is not None:
            needed = max(needed, self.vol_lookback)
        return needed + 1 + self.execution_lag_days


@dataclass
class BacktestResult:
    config: StrategyConfig
    close: pd.Series            # full price history
    fast_ma: pd.Series
    slow_ma: pd.Series
    raw_signal: pd.Series
    target: pd.Series
    held: pd.Series
    returns: pd.Series          # strategy daily returns, net of costs (evaluation window)
    benchmark_returns: pd.Series
    equity: pd.Series           # evaluation window, starts at initial_capital
    benchmark_equity: pd.Series
    drawdown: pd.Series         # fractions, <= 0
    benchmark_drawdown: pd.Series
    trades: pd.DataFrame
    metrics: dict
    benchmark_metrics: dict


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def moving_average(close: pd.Series, window: int, kind: str = "sma") -> pd.Series:
    if kind == "sma":
        return close.rolling(window, min_periods=window).mean()
    if kind == "ema":
        return close.ewm(span=window, adjust=False, min_periods=window).mean()
    raise ValueError(f"unknown moving average type: {kind}")


def _apply_stop_loss(raw: np.ndarray, close: np.ndarray, stop_frac: float) -> np.ndarray:
    """Force the signal flat after a close-based stop, until the raw signal resets.

    Path dependent, so it needs a loop. It only uses closes up to day t when it
    decides day t, so there is no look-ahead. The entry reference is the close
    of the signal day (the real fill happens later, so this is an approximation).
    A stop can only trigger on a close; intraday spikes through the stop are not seen.
    """
    if stop_frac <= 0:
        return raw.copy()
    out = raw.copy()
    entry = np.nan
    locked = False
    for t in range(len(raw)):
        if raw[t] == 0:
            locked, entry, out[t] = False, np.nan, 0.0
            continue
        if locked:
            out[t] = 0.0
            continue
        if np.isnan(entry):
            entry = close[t]
        if close[t] <= entry * (1.0 - stop_frac):
            locked, entry, out[t] = True, np.nan, 0.0
        else:
            out[t] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #
def build_positions(close: pd.Series, cfg: StrategyConfig):
    ret = close.pct_change()
    fast_ma = moving_average(close, cfg.fast, cfg.ma_type)
    slow_ma = moving_average(close, cfg.slow, cfg.ma_type)

    # Comparisons with NaN are False, so the warm-up period is automatically flat.
    raw = (fast_ma > slow_ma).astype(float)

    signal = pd.Series(
        _apply_stop_loss(raw.to_numpy(), close.to_numpy(), cfg.stop_loss_pct / 100.0),
        index=close.index,
    )

    if cfg.target_vol is None:
        size = pd.Series(1.0, index=close.index)
    else:
        realized = ret.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback).std() \
            * np.sqrt(cfg.periods_per_year)
        # Shrink the position when the market is wild; stay flat until vol is known.
        size = (cfg.target_vol / realized).clip(upper=cfg.max_leverage).fillna(0.0)

    target = signal * size
    held = target.shift(1 + cfg.execution_lag_days).fillna(0.0)
    return ret, fast_ma, slow_ma, raw, target, held


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _clean(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def performance_metrics(returns: pd.Series, risk_free: float = 0.0,
                        periods_per_year: int = 252) -> dict:
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return {}
    ppy = periods_per_year
    equity = (1.0 + r).cumprod()
    years = n / ppy
    total_return = equity.iloc[-1] - 1.0
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if equity.iloc[-1] > 0 else -1.0

    std = r.std(ddof=1)
    ann_vol = std * np.sqrt(ppy)

    excess = r - risk_free / ppy
    sharpe = excess.mean() / std * np.sqrt(ppy) if std > 0 else np.nan

    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean()) * np.sqrt(ppy)
    sortino = excess.mean() * ppy / downside if downside > 0 else np.nan

    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    # Standard error of the annualised Sharpe (Lo 2002, iid returns).
    sr_per_period = excess.mean() / std if std > 0 else np.nan
    sharpe_se = np.sqrt((1.0 + 0.5 * sr_per_period ** 2) / years) if std > 0 else np.nan

    return {
        "total_return": _clean(total_return),
        "cagr": _clean(cagr),
        "ann_vol": _clean(ann_vol),
        "sharpe": _clean(sharpe),
        "sharpe_se": _clean(sharpe_se),
        "sortino": _clean(sortino),
        "max_drawdown": _clean(max_dd),
        "calmar": _clean(calmar),
        "years": _clean(years),
    }


# --------------------------------------------------------------------------- #
# Trades
# --------------------------------------------------------------------------- #
TRADE_COLUMNS = ["entry_date", "entry_price", "exit_date", "exit_price",
                 "avg_size", "return_pct", "pnl", "bars", "exit_reason"]


def extract_trades(close, held, raw, strat_ret_full, equity_full, cfg) -> pd.DataFrame:
    """One row per contiguous run of held > 0.

    held[i] > 0 means the order was filled at close[i-1], because held[i] earns
    return[i] = close[i] / close[i-1] - 1. The return of a trade is the compounded
    net strategy return over the run, including the exit-day cost.
    """
    h = held.to_numpy()
    n = len(h)
    rows = []
    i = 0
    while i < n:
        if h[i] <= 0:
            i += 1
            continue
        j = i
        while j < n and h[j] > 0:
            j += 1                                   # run is [i, j)
        is_open = j >= n
        last = n - 1 if is_open else j               # include exit-day cost
        trade_ret = float((1.0 + strat_ret_full.iloc[i:last + 1]).prod() - 1.0)
        entry_exec = i - 1
        exit_exec = n - 1 if is_open else j - 1
        equity_before = float(equity_full.iloc[i - 1])

        if is_open:
            reason = "Open (marked to market)"
        else:
            decision_day = j - 1 - cfg.execution_lag_days   # day target first went flat
            reason = "Stop Loss" if raw.iloc[decision_day] == 1 else "Signal"

        rows.append({
            "entry_date": close.index[entry_exec],
            "entry_price": float(close.iloc[entry_exec]),
            "exit_date": close.index[exit_exec],
            "exit_price": float(close.iloc[exit_exec]),
            "avg_size": float(h[i:j].mean()),
            "return_pct": trade_ret * 100.0,
            "pnl": equity_before * trade_ret,
            "bars": exit_exec - entry_exec,
            "exit_reason": reason,
        })
        i = j
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def trade_statistics(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": None, "avg_win_pct": None,
                "avg_loss_pct": None, "profit_factor": None}
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    gross_loss = abs(losses["pnl"].sum())
    return {
        "n_trades": n,
        "win_rate": len(wins) / n,
        "avg_win_pct": _clean(wins["return_pct"].mean()) if len(wins) else None,
        "avg_loss_pct": _clean(losses["return_pct"].mean()) if len(losses) else None,
        "profit_factor": _clean(wins["pnl"].sum() / gross_loss) if gross_loss > 0 else None,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_backtest(close: pd.Series, cfg: StrategyConfig) -> BacktestResult:
    cfg.validate()
    close = close.astype(float).dropna()
    if close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError("price index must be unique and sorted oldest to newest")
    if (close <= 0).any():
        raise ValueError("prices must be positive")
    if len(close) < cfg.warmup + 30:
        raise ValueError(f"Not enough data: need at least {cfg.warmup + 30} bars, got {len(close)}")

    ret, fast_ma, slow_ma, raw, target, held = build_positions(close, cfg)

    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * cfg.cost_bps / 1e4
    strat_ret_full = held * ret.fillna(0.0) - cost

    equity_full = cfg.initial_capital * (1.0 + strat_ret_full).cumprod()
    trades = extract_trades(close, held, raw, strat_ret_full, equity_full, cfg)

    # Evaluate strategy and benchmark over the same window: from the first bar
    # that could hold a position. Otherwise the warm-up flatters or hurts one side.
    w = cfg.warmup
    r = strat_ret_full.iloc[w:]
    b = ret.fillna(0.0).iloc[w:]
    equity = cfg.initial_capital * (1.0 + r).cumprod()
    bench_equity = cfg.initial_capital * (1.0 + b).cumprod()

    metrics = performance_metrics(r, cfg.risk_free, cfg.periods_per_year)
    metrics.update(trade_statistics(trades))
    metrics["exposure"] = _clean((held.iloc[w:] > 0).mean())
    metrics["avg_position"] = _clean(held.iloc[w:].mean())
    metrics["total_cost_pct"] = _clean(cost.iloc[w:].sum())      # additive approximation
    metrics["turnover_per_year"] = _clean(
        turnover.iloc[w:].sum() / (len(r) / cfg.periods_per_year))
    metrics["initial_capital"] = cfg.initial_capital
    metrics["final_value"] = float(equity.iloc[-1])

    bench_metrics = performance_metrics(b, cfg.risk_free, cfg.periods_per_year)

    return BacktestResult(
        config=cfg, close=close, fast_ma=fast_ma, slow_ma=slow_ma, raw_signal=raw,
        target=target, held=held, returns=r, benchmark_returns=b,
        equity=equity, benchmark_equity=bench_equity,
        drawdown=equity / equity.cummax() - 1.0,
        benchmark_drawdown=bench_equity / bench_equity.cummax() - 1.0,
        trades=trades, metrics=metrics, benchmark_metrics=bench_metrics,
    )
