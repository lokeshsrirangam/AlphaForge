"""Walk-forward validation tests.

The three that matter most:

  - no look-ahead: prices after a fold's train window cannot change the
    parameters that fold picks
  - window hygiene: test windows are disjoint and each starts strictly after
    its own train window
  - null test: on driftless random walks the stitched out-of-sample Sharpe is
    about zero, i.e. the machinery does not manufacture an edge
"""
import numpy as np
import pandas as pd

from alphaforge import StrategyConfig, synthetic_prices
from alphaforge.walkforward import (
    DEFAULT_TEST_LEN,
    DEFAULT_TRAIN_LEN,
    build_grid,
    run_fold,
    select_params,
    walk_forward,
)


def _cfg(**kw):
    """Base config for tests: no vol target, no costs unless asked."""
    base = dict(fast=10, slow=30, target_vol=None, cost_bps=0.0,
                execution_lag_days=1)
    base.update(kw)
    return StrategyConfig(**base)


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def test_grid_is_fixed_and_drops_fast_ge_slow():
    """The grid is exactly the pre-agreed one, and fast >= slow never appears."""
    grid = build_grid()
    assert grid == [
        {"fast": 5, "slow": 50},
        {"fast": 10, "slow": 50},
        {"fast": 20, "slow": 50},
        {"fast": 5, "slow": 100},
        {"fast": 10, "slow": 100},
        {"fast": 20, "slow": 100},
        {"fast": 50, "slow": 100},
        {"fast": 5, "slow": 200},
        {"fast": 10, "slow": 200},
        {"fast": 20, "slow": 200},
        {"fast": 50, "slow": 200},
    ]
    assert all(p["fast"] < p["slow"] for p in grid)
    # The grid is the same every time it is built.
    assert grid == build_grid()


def test_select_params_ties_go_to_first_grid_entry():
    """When every combination scores the same, the first grid entry wins."""
    close = synthetic_prices(days=600, seed=7)
    grid = build_grid()
    flat = pd.Series(100.0, index=close.index)      # no movement at all

    params, sharpe = select_params(flat, 250, 500, grid, _cfg())
    assert params == grid[0]
    assert sharpe is None      # a flat series has no Sharpe


# --------------------------------------------------------------------------- #
# No look-ahead
# --------------------------------------------------------------------------- #
def test_prices_after_train_window_do_not_change_chosen_params():
    """Changing prices after a fold's train window leaves that fold's choice alone.

    This is the single most important property of walk-forward validation: the
    parameters for a fold must be a function of the train window only.
    """
    close = synthetic_prices(days=3000, seed=42)
    grid = build_grid()
    cfg = _cfg(cost_bps=10.0)

    train_start, train_end = 300, 300 + 756 - 1
    test_start, test_end = train_end + 1, train_end + 252

    chosen_clean, _ = select_params(close, train_start, train_end, grid, cfg)
    fold_clean = run_fold(close, train_start, train_end, test_start, test_end,
                          grid, cfg)

    # Scramble everything after the train window.
    tampered = close.copy()
    tampered.iloc[train_end + 1:] = tampered.iloc[train_end + 1:] * 2.5

    chosen_tampered, _ = select_params(tampered, train_start, train_end, grid, cfg)
    fold_tampered = run_fold(tampered, train_start, train_end, test_start, test_end,
                             grid, cfg)

    assert chosen_clean == chosen_tampered
    assert fold_clean.chosen_params == fold_tampered.chosen_params
    assert fold_clean.is_sharpe == fold_tampered.is_sharpe


def test_walk_forward_earlier_folds_ignore_later_data():
    """Editing the tail of the series cannot change an early fold's choice."""
    close = synthetic_prices(days=3000, seed=11)
    grid = build_grid()
    cfg = _cfg(cost_bps=10.0)

    res_a = walk_forward(close, cfg, train_len=756, test_len=252)
    tampered = close.copy()
    tampered.iloc[-500:] = tampered.iloc[-500:] * 3.0
    res_b = walk_forward(tampered, cfg, train_len=756, test_len=252)

    assert res_a.n_folds > 1
    # The last fold may legitimately change; the earlier ones must not.
    for a, b in zip(res_a.folds[:-1], res_b.folds[:-1]):
        assert a.chosen_params == b.chosen_params


# --------------------------------------------------------------------------- #
# Window hygiene
# --------------------------------------------------------------------------- #
def test_test_windows_are_disjoint_and_start_after_train_windows():
    """Test windows never overlap, and each starts strictly after its train window."""
    close = synthetic_prices(days=3000, seed=42)
    res = walk_forward(close, _cfg(cost_bps=10.0),
                       train_len=DEFAULT_TRAIN_LEN, test_len=DEFAULT_TEST_LEN)

    assert res.n_folds > 1
    previous_test_end = None
    for fold in res.folds:
        assert fold.test_start > fold.train_end, \
            f"fold {fold.index}: test starts {fold.test_start} but train ends {fold.train_end}"
        if previous_test_end is not None:
            assert fold.test_start > previous_test_end, \
                f"fold {fold.index}: test window overlaps the previous one"
        previous_test_end = fold.test_end


def test_stitched_oos_returns_are_in_order_and_non_overlapping():
    """The stitched series contains each test bar exactly once, in order."""
    close = synthetic_prices(days=3000, seed=5)
    res = walk_forward(close, _cfg(cost_bps=10.0), train_len=756, test_len=252)

    assert res.oos_returns.index.is_monotonic_increasing
    assert res.oos_returns.index.is_unique
    assert len(res.oos_returns) == sum(len(f.test_returns) for f in res.folds)
    assert res.oos_returns.index[0] == res.folds[0].test_start
    assert res.oos_returns.index[-1] == res.folds[-1].test_end


def test_default_lengths_are_756_and_252():
    """The documented defaults are the ones actually used."""
    assert DEFAULT_TRAIN_LEN == 756
    assert DEFAULT_TEST_LEN == 252
    close = synthetic_prices(days=3000, seed=3)
    res = walk_forward(close, _cfg())
    for fold in res.folds:
        assert len(fold.test_returns) == 252


def test_step_smaller_than_test_len_is_rejected():
    """Overlapping test windows are refused rather than silently produced."""
    close = synthetic_prices(days=2000, seed=3)
    try:
        walk_forward(close, _cfg(), train_len=500, test_len=200, step=100)
    except ValueError:
        return
    raise AssertionError("step < test_len should raise ValueError")


# --------------------------------------------------------------------------- #
# Per-fold and headline output
# --------------------------------------------------------------------------- #
def test_folds_report_the_required_fields():
    """Each fold reports params, in-sample and OOS Sharpe, returns and n_tried."""
    close = synthetic_prices(days=3000, seed=42)
    res = walk_forward(close, _cfg(cost_bps=10.0), train_len=756, test_len=252)

    grid = build_grid()
    for fold in res.folds:
        assert fold.fast in (5, 10, 20, 50)
        assert fold.slow in (50, 100, 200)
        assert fold.fast < fold.slow
        assert fold.n_tried == len(grid)
        assert len(fold.test_returns) == 252
        assert fold.oos_sharpe is None or np.isfinite(fold.oos_sharpe)


def test_headline_metrics_come_from_the_stitched_series():
    """The headline Sharpe/CAGR/max-drawdown are computed on one stitched series."""
    close = synthetic_prices(days=3000, seed=42)
    res = walk_forward(close, _cfg(cost_bps=10.0), train_len=756, test_len=252)

    for key in ("sharpe", "sharpe_se", "cagr", "max_drawdown"):
        assert key in res.oos_metrics
        assert key in res.benchmark_metrics

    # Buy and hold covers exactly the same dates as the strategy.
    assert list(res.benchmark_returns.index) == list(res.oos_returns.index)

    # The headline is not the mean of the per-window Sharpes (that is secondary).
    assert res.mean_fold_oos_sharpe is not None
    assert "mean_fold_oos_sharpe" in res.to_dict()


# --------------------------------------------------------------------------- #
# Null test
# --------------------------------------------------------------------------- #
def test_null_test_driftless_random_walks_have_no_edge():
    """On driftless random walks the mean stitched OOS Sharpe is about zero.

    100+ independent seeds with zero costs. If the walk-forward machinery were
    leaking future information or manufacturing an edge, this mean would be
    clearly positive.
    """
    sharpes = []
    n_seeds = 120
    for seed in range(n_seeds):
        close = synthetic_prices(days=2400, seed=seed, daily_drift=0.0)
        res = walk_forward(close, _cfg(cost_bps=0.0), train_len=756, test_len=252)
        sharpe = res.oos_metrics.get("sharpe")
        if sharpe is not None and np.isfinite(sharpe):
            sharpes.append(sharpe)

    assert len(sharpes) >= 100, f"only {len(sharpes)} usable seeds"
    mean_sharpe = float(np.mean(sharpes))
    assert abs(mean_sharpe) < 0.2, \
        (f"mean stitched OOS Sharpe over {len(sharpes)} driftless seeds = {mean_sharpe:.4f}; "
         "expected about 0 (no manufactured edge)")
