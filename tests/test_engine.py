"""Run with:  python -m pytest -q

These tests exist to catch the mistakes that make backtests lie:
look-ahead, free trading, and results that do not reconcile.
"""
import numpy as np
import pandas as pd

from alphaforge import StrategyConfig, run_backtest, synthetic_prices
from alphaforge.engine import build_positions, performance_metrics


def _cfg(**kw):
    base = dict(fast=10, slow=30, target_vol=None, cost_bps=0.0, execution_lag_days=1)
    base.update(kw)
    return StrategyConfig(**base)


def test_no_lookahead_future_prices_cannot_change_the_past():
    close = synthetic_prices(days=600, seed=1)
    cfg = _cfg(target_vol=0.15, stop_loss_pct=5.0, cost_bps=10.0)
    k = 400
    tampered = close.copy()
    tampered.iloc[k + 1:] = tampered.iloc[k + 1:] * 3.0      # wreck the future

    a = run_backtest(close, cfg)
    b = run_backtest(tampered, cfg)

    full = a.config  # same config
    held_a = build_positions(close, full)[5]
    held_b = build_positions(tampered, full)[5]
    pd.testing.assert_series_equal(held_a.iloc[:k + 1], held_b.iloc[:k + 1])
    pd.testing.assert_series_equal(a.returns.loc[:close.index[k]], b.returns.loc[:close.index[k]])


def test_same_close_execution_is_more_optimistic_than_next_close_on_trending_data():
    # On a smooth trend, trading one day late must not look better than trading on time.
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(100 * np.exp(np.linspace(0, 1, 300)), index=idx)
    now = run_backtest(close, _cfg(execution_lag_days=0)).metrics["total_return"]
    late = run_backtest(close, _cfg(execution_lag_days=1)).metrics["total_return"]
    assert now >= late


def test_costs_reduce_returns():
    close = synthetic_prices(days=1500, seed=3, daily_drift=0.0003)
    free = run_backtest(close, _cfg(cost_bps=0)).metrics["total_return"]
    costly = run_backtest(close, _cfg(cost_bps=50)).metrics["total_return"]
    assert costly < free


def test_always_long_matches_buy_and_hold_without_costs():
    idx = pd.bdate_range("2020-01-01", periods=200)
    close = pd.Series(100 * np.exp(np.linspace(0, 0.5, 200)), index=idx)   # up every day
    res = run_backtest(close, _cfg())
    # Once the strategy is long it never leaves, so over the evaluation window it is
    # fully invested from (almost) the first bar: returns match the benchmark.
    assert abs(res.metrics["total_return"] - res.benchmark_metrics["total_return"]) < 0.02


def test_stop_loss_exits_a_falling_position_early():
    # Flat, then a rally that triggers a long, then a fast crash followed by a slide.
    # The stop is measured from the entry price, so it fires while the (slower)
    # moving averages still say "long". Without the stop we ride the slide down.
    prices = np.concatenate([np.full(40, 100.0), np.linspace(100, 130, 10), [112.0, 95.0],
                             np.linspace(95, 70, 20), np.full(20, 70.0)])
    close = pd.Series(prices, index=pd.bdate_range("2020-01-01", periods=len(prices)))
    with_stop = run_backtest(close, _cfg(stop_loss_pct=5.0))
    without = run_backtest(close, _cfg(stop_loss_pct=0.0))
    assert list(with_stop.trades["exit_reason"]) == ["Stop Loss"]
    assert list(without.trades["exit_reason"]) == ["Signal"]
    assert with_stop.metrics["max_drawdown"] > without.metrics["max_drawdown"]   # less negative


def test_trades_reconcile_with_equity_curve():
    close = synthetic_prices(days=1200, seed=7, daily_drift=0.0002)
    res = run_backtest(close, _cfg(target_vol=0.15, cost_bps=15.0))
    compounded = float(np.prod(1 + res.trades["return_pct"] / 100.0))
    growth = res.equity.iloc[-1] / res.equity.iloc[0]
    # equity.iloc[0] already includes day-one return; tolerance covers that single bar.
    assert abs(compounded - growth) < 1e-3
    assert len(res.trades) > 0


def test_volatility_targeting_reduces_volatility():
    close = synthetic_prices(days=2000, seed=11, daily_vol=0.02, daily_drift=0.0003)
    plain = run_backtest(close, _cfg(target_vol=None)).metrics["ann_vol"]
    targeted = run_backtest(close, _cfg(target_vol=0.10)).metrics["ann_vol"]
    assert targeted < plain


def test_metrics_match_hand_calculation():
    r = pd.Series([0.01, -0.01, 0.02, -0.005, 0.0, 0.015] * 50)
    m = performance_metrics(r, risk_free=0.0, periods_per_year=252)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert abs(m["sharpe"] - expected) < 1e-9
    assert m["max_drawdown"] <= 0


def test_null_test_no_edge_on_random_walk():
    """On a driftless random walk a trend rule has no edge: mean Sharpe ~ 0.

    This is the sanity check that tells us the engine is not manufacturing alpha.
    """
    sharpes = []
    for seed in range(150):
        close = synthetic_prices(days=2000, seed=seed, daily_drift=0.0)
        m = run_backtest(close, _cfg()).metrics
        if m.get("sharpe") is not None:
            sharpes.append(m["sharpe"])
    assert abs(np.mean(sharpes)) < 0.15


def test_synthetic_prices_are_reproducible():
    a = synthetic_prices(days=100, seed=5)
    b = synthetic_prices(days=100, seed=5)
    pd.testing.assert_series_equal(a, b)
