"""Walk-forward validation: out-of-sample testing of the moving-average strategy.

Pure module. No Flask, no file I/O, no network. It takes a price series and a
base :class:`StrategyConfig` and produces honest out-of-sample numbers.

What walk-forward validation does
---------------------------------
1. The data is cut into rolling train windows and non-overlapping test windows.
2. On each train window, the parameter pair (fast, slow) is chosen from a grid
   that is fixed in advance. The train window is the only data the chooser sees.
3. The chosen parameters are then run on the *next* test window, which the
   chooser never saw.
4. The out-of-sample returns of every fold are stitched into one series and
   scored as a single backtest. That stitched score is the headline number.

The grid never changes after seeing results. If the strategy has no edge, the
stitched out-of-sample Sharpe will hover around zero (or below, because picking
the best of several parameter sets on past data is itself a small penalty).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from .engine import StrategyConfig, performance_metrics, run_backtest

# --------------------------------------------------------------------------- #
# The grid is fixed in advance and is never adjusted after seeing results.
# --------------------------------------------------------------------------- #
GRID_SLOW: tuple[int, ...] = (50, 100, 200)
GRID_FAST: tuple[int, ...] = (5, 10, 20, 50)

DEFAULT_TRAIN_LEN = 756          # ~3 years of daily bars
DEFAULT_TEST_LEN = 252           # ~1 year of daily bars


def build_grid(slow_values: Sequence[int] = GRID_SLOW,
               fast_values: Sequence[int] = GRID_FAST) -> list[dict[str, int]]:
    """The fixed parameter grid.

    Slow windows in ascending order, then fast windows in ascending order.
    Combinations where ``fast >= slow`` are dropped (a crossover needs the
    fast window to be the shorter one). Ties during selection are broken by
    this order: the earliest entry wins.
    """
    grid: list[dict[str, int]] = []
    for slow in slow_values:
        for fast in fast_values:
            if fast < slow:
                grid.append({"fast": int(fast), "slow": int(slow)})
    return grid


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class Fold:
    """One train/test round of the walk-forward run."""

    index: int                       # 1-based fold number
    fast: int
    slow: int
    is_sharpe: float | None          # in-sample Sharpe on the train window
    oos_sharpe: float | None         # out-of-sample Sharpe on the test window
    test_returns: pd.Series          # daily OOS returns for this test window
    n_tried: int                     # number of grid combinations evaluated
    train_start: pd.Timestamp
    train_end: pd.Timestamp          # last bar of the train window
    test_start: pd.Timestamp         # first bar of the test window
    test_end: pd.Timestamp           # last bar of the test window

    @property
    def chosen_params(self) -> dict[str, int]:
        return {"fast": self.fast, "slow": self.slow}

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "fast": self.fast,
            "slow": self.slow,
            "is_sharpe": self.is_sharpe,
            "oos_sharpe": self.oos_sharpe,
            "n_tried": self.n_tried,
            "train_start": self.train_start.strftime("%Y-%m-%d"),
            "train_end": self.train_end.strftime("%Y-%m-%d"),
            "test_start": self.test_start.strftime("%Y-%m-%d"),
            "test_end": self.test_end.strftime("%Y-%m-%d"),
        }


@dataclass
class WalkForwardResult:
    """Everything the walk-forward run produced."""

    folds: list[Fold]
    oos_returns: pd.Series           # ONE stitched out-of-sample return series
    oos_metrics: dict                # performance_metrics() on the stitched series
    benchmark_returns: pd.Series     # buy and hold over the same OOS dates
    benchmark_metrics: dict
    mean_fold_oos_sharpe: float | None    # secondary statistic only
    mean_fold_is_sharpe: float | None    # secondary statistic only
    param_counts: dict[tuple[int, int], int]
    grid: list[dict[str, int]]

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def to_dict(self) -> dict:
        return {
            "folds": [f.to_dict() for f in self.folds],
            "oos_metrics": self.oos_metrics,
            "benchmark_metrics": self.benchmark_metrics,
            "mean_fold_oos_sharpe": self.mean_fold_oos_sharpe,
            "mean_fold_is_sharpe": self.mean_fold_is_sharpe,
            "param_counts": [{"fast": f, "slow": s, "count": c}
                             for (f, s), c in self.param_counts.items()],
            "grid": self.grid,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _candidate_cfg(base_cfg: StrategyConfig, params: dict[str, int]) -> StrategyConfig:
    """A copy of the base config with fast/slow replaced by one grid entry.

    Every other setting (costs, vol target, stop, execution lag) is inherited,
    so a grid comparison is like for like.
    """
    return replace(base_cfg, fast=int(params["fast"]), slow=int(params["slow"]))


def _windowed_returns(close_slice: pd.Series, cfg: StrategyConfig) -> pd.Series:
    """Strategy returns for a price slice, aligned to the slice's own index.

    ``run_backtest`` only reports the part of the series after the indicator
    warm-up. We rebuild a full-length series (NaN during the warm-up) so the
    caller can cut out exactly the window it wants, whatever the warm-up is.
    """
    res = run_backtest(close_slice, cfg)
    out = pd.Series(np.nan, index=res.close.index, dtype=float)
    out.iloc[cfg.warmup:] = res.returns.to_numpy()
    return out


def _sharpe(returns: pd.Series, cfg: StrategyConfig) -> float | None:
    """Annualised Sharpe of a return series, or None when it cannot be computed."""
    metrics = performance_metrics(returns, cfg.risk_free, cfg.periods_per_year)
    value = metrics.get("sharpe")
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _mean_or_none(values: list[float]) -> float | None:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


# --------------------------------------------------------------------------- #
# Parameter selection
# --------------------------------------------------------------------------- #
def select_params(close: pd.Series, train_start: int, train_end: int,
                  grid: list[dict[str, int]], base_cfg: StrategyConfig
                  ) -> tuple[dict[str, int], float | None]:
    """Pick (fast, slow) using the train window only.

    Each grid entry is backtested on the train window (net of costs) and the
    one with the highest in-sample Sharpe wins. A NaN Sharpe counts as -inf, so
    a combination that cannot even be scored never wins. Ties go to whichever
    combination appears first in the grid.

    Args:
        close: the full price series.
        train_start: first bar of the train window (inclusive).
        train_end: last bar of the train window (inclusive).
        grid: the fixed parameter grid.
        base_cfg: config template; fast/slow are replaced per candidate.

    Returns:
        (chosen params, chosen in-sample Sharpe or None)
    """
    candidates = [_candidate_cfg(base_cfg, p) for p in grid]
    warmup = max(c.warmup for c in candidates) if candidates else base_cfg.warmup
    slice_start = max(0, train_start - warmup)
    train_slice = close.iloc[slice_start:train_end + 1]
    if len(train_slice) == 0:
        raise ValueError("Empty train window")

    best_params: dict[str, int] | None = None
    best_score = -np.inf
    best_sharpe: float | None = None

    for params, cfg in zip(grid, candidates):
        try:
            full = _windowed_returns(train_slice, cfg)
            window = full.iloc[train_start - slice_start: train_end - slice_start + 1]
            score = _sharpe(window, cfg)
        except Exception:
            score = None

        # A Sharpe that cannot be computed counts as -inf (worst possible).
        comparable = score if (score is not None and np.isfinite(score)) else -np.inf

        # Strictly greater, so a tie keeps the earlier grid entry. The first
        # entry is picked up by `>=` only when nothing scored at all, so a
        # genuinely unscorable grid still returns a usable (if arbitrary) pair.
        if comparable > best_score or (best_params is None):
            best_score = comparable
            best_params = params
            best_sharpe = score

    if best_params is None:
        raise ValueError("Empty parameter grid")
    return best_params, best_sharpe


# --------------------------------------------------------------------------- #
# One fold
# --------------------------------------------------------------------------- #
def run_fold(close: pd.Series, train_start: int, train_end: int,
             test_start: int, test_end: int, grid: list[dict[str, int]],
             base_cfg: StrategyConfig, index: int = 1) -> Fold:
    """Select parameters on the train window, then test them on the next window.

    The test window is run on a slice that starts ``cfg.warmup`` bars *before*
    the test window, so the moving averages and the volatility estimate are
    already populated on the first test bar. Only the returns from the test
    start onward are kept. Nothing after the test window is read.
    """
    params, is_sharpe = select_params(close, train_start, train_end, grid, base_cfg)
    cfg = _candidate_cfg(base_cfg, params)

    warmup = cfg.warmup
    slice_start = max(0, test_start - warmup)
    test_slice = close.iloc[slice_start: test_end + 1]
    if len(test_slice) < warmup + 2:
        raise ValueError("Test window too short for the chosen parameters")

    full = _windowed_returns(test_slice, cfg)
    test_returns = full.iloc[test_start - slice_start: test_end - slice_start + 1]

    oos_sharpe = _sharpe(test_returns, cfg)

    return Fold(
        index=index,
        fast=params["fast"],
        slow=params["slow"],
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        test_returns=test_returns,
        n_tried=len(grid),
        train_start=close.index[train_start],
        train_end=close.index[train_end],
        test_start=close.index[test_start],
        test_end=close.index[test_end],
    )


# --------------------------------------------------------------------------- #
# The whole walk
# --------------------------------------------------------------------------- #
def walk_forward(close: pd.Series, base_cfg: StrategyConfig,
                 train_len: int = DEFAULT_TRAIN_LEN,
                 test_len: int = DEFAULT_TEST_LEN,
                 step: int | None = None,
                 grid: list[dict[str, int]] | None = None) -> WalkForwardResult:
    """Run the full walk-forward validation.

    Defaults: 756-bar train windows, 252-bar test windows, stepping by the
    test length so the test windows are back to back and never overlap. The
    first train window starts after enough bars for the slowest indicator, so
    every window has a full warm-up behind it.

    Args:
        close: price series, oldest first.
        base_cfg: config template. ``fast``/``slow`` are replaced per candidate;
            every other setting (costs, vol target, stop, execution lag) is
            inherited so the run matches the single-backtest settings.
        train_len: length of the train window in bars.
        test_len: length of the test window in bars.
        step: how far the window moves forward each fold. Defaults to
            ``test_len``. Must be at least ``test_len``, otherwise test
            windows would overlap.
        grid: the parameter grid. Defaults to the fixed grid.

    Returns:
        A :class:`WalkForwardResult` holding the folds, the single stitched
        out-of-sample return series, and the metrics computed on it.
    """
    if train_len <= 0 or test_len <= 0:
        raise ValueError("train_len and test_len must be positive")
    if step is None:
        step = test_len
    if step < test_len:
        raise ValueError("step must be >= test_len, otherwise test windows overlap")
    if grid is None:
        grid = build_grid()
    if not grid:
        raise ValueError("Empty parameter grid")

    close = pd.Series(close, dtype=float).dropna()
    if close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError("price index must be unique and sorted oldest to newest")
    if (close <= 0).any():
        raise ValueError("prices must be positive")

    # The slowest indicator in the grid decides how much history each window
    # needs behind it before it can be scored.
    warmup = max(_candidate_cfg(base_cfg, p).warmup for p in grid)

    folds: list[Fold] = []
    train_start = warmup
    fold_no = 0

    while True:
        train_end = train_start + train_len - 1
        test_start = train_end + 1              # strictly after the train window
        test_end = test_start + test_len - 1
        if test_end >= len(close):
            break
        fold_no += 1
        folds.append(run_fold(close, train_start, train_end, test_start, test_end,
                              grid, base_cfg, index=fold_no))
        train_start += step

    if folds:
        oos_returns = pd.concat([f.test_returns for f in folds])
        oos_returns = oos_returns[oos_returns.notna()]
    else:
        oos_returns = pd.Series(dtype=float)

    oos_metrics = performance_metrics(oos_returns, base_cfg.risk_free,
                                       base_cfg.periods_per_year)

    # Buy and hold over exactly the same out-of-sample dates.
    bench = close.pct_change()
    benchmark_returns = bench.reindex(oos_returns.index) if len(oos_returns) else pd.Series(dtype=float)
    benchmark_metrics = performance_metrics(benchmark_returns, base_cfg.risk_free,
                                            base_cfg.periods_per_year)

    param_counts: dict[tuple[int, int], int] = {}
    for f in folds:
        key = (f.fast, f.slow)
        param_counts[key] = param_counts.get(key, 0) + 1

    return WalkForwardResult(
        folds=folds,
        oos_returns=oos_returns,
        oos_metrics=oos_metrics,
        benchmark_returns=benchmark_returns,
        benchmark_metrics=benchmark_metrics,
        mean_fold_oos_sharpe=_mean_or_none([f.oos_sharpe for f in folds]),
        mean_fold_is_sharpe=_mean_or_none([f.is_sharpe for f in folds]),
        param_counts=param_counts,
        grid=grid,
    )
