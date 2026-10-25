# AlphaForge

A backtesting tool for a moving-average trend strategy, built to answer one question honestly: would this rule have made money **after costs**, and is the result distinguishable from luck?

Built with Python, pandas, NumPy, Flask and Plotly.

## What it does

- Loads adjusted daily prices from Yahoo Finance (cached on disk)
- SMA or EMA crossover candidates, with optional close-based stop-loss
- Close-based Donchian breakout strategy with prior-close entry and exit bands
- Volatility-targeted position sizing
- Trading costs in basis points on every change in position
- Next-close execution by default (same-close available, flagged as optimistic)
- Metrics: CAGR, volatility, Sharpe (with standard error), Sortino, max drawdown, Calmar, time in market, cost drag, trade statistics
- Strategy vs buy & hold over the same window
- Monthly return calendar built from compounded net daily returns
- Risk-on / risk-off attribution using a prior-close 200-session moving average
- Walk-forward validation: parameters are chosen on each training window and scored on the held-out test window that follows it
- Warnings when a result is statistically weak (few trades, short sample, Sharpe within noise of zero, suspiciously high Sharpe)

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python app.py        # http://127.0.0.1:5000
```

## Methodology

```
day t    : see close[t], compute the signal, decide the target position
day t+1  : order executes at close[t+1]
day t+2  : the position earns close[t+2] / close[t+1] - 1
```

Tests in `tests/test_engine.py` check the things that make backtests lie:

- changing future prices cannot change past positions (no look-ahead)
- costs always reduce returns
- trade returns reconcile with the equity curve
- on driftless random walks the average Sharpe is about zero (the engine does not manufacture edge)

## Walk-forward validation

`alphaforge/walkforward.py` asks the harder question: would the parameter choice
have worked on data it had never seen?

- The grid is fixed in advance: slow ∈ [50, 100, 200], fast ∈ [5, 10, 20, 50],
  keeping only fast < slow. It is never changed after seeing results.
- Each fold picks (fast, slow) on the training window alone, by in-sample Sharpe
  net of costs. A Sharpe that cannot be computed counts as -inf; ties go to the
  first entry in the grid.
- The chosen pair is then run on the next test window, on a slice that starts
  `cfg.warmup` bars earlier so the indicators are already warm, keeping only
  the returns from the test start onward. Nothing after the test window is read.
- Defaults: 756-bar train windows, 252-bar test windows, stepping by the test
  length, so test windows are back to back and never overlap.
- The headline result is **one stitched out-of-sample return series**, scored
  with the same `performance_metrics()` as everything else (Sharpe with standard
  error, CAGR, max drawdown) and shown next to buy and hold over the same
  out-of-sample dates. The mean of the per-window Sharpes is secondary only.

Tests in `tests/test_walkforward.py` check the things that make walk-forward
validation lie:

- changing prices after a fold's train window cannot change that fold's choice
- test windows are disjoint and each starts strictly after its own train window
- on 120 seeds of driftless random walks with zero costs, the mean stitched
  out-of-sample Sharpe is below 0.2 in absolute value

### Out-of-sample result (SPY, 2012-01-01 to 2025-07-22)

Defaults, 10 bps costs, 15% vol target, no stop, next-close execution.
9 folds, out-of-sample period 2015-10-22 to 2024-10-25 (2268 bars).

|                     | Strategy | Buy & hold |
| ------------------- | -------: | ---------: |
| Total return        |  86.81%  |   235.40%  |
| CAGR                |   7.19%  |    14.39%  |
| Sharpe              |   0.70   |     0.84   |
| Sharpe std. error   |  ± 0.33  |   ± 0.33   |
| Max drawdown        | -19.49%  |  -33.72%  |

Read plainly: **the strategy did not beat buy and hold.** It made less money
and had a lower risk-adjusted return; what it gave up in return was a much
smaller drawdown (-19% vs -34%). The out-of-sample Sharpe of 0.70 is barely
above the 0.33 standard error, so even the smaller drawdown is weak evidence.
The parameter search is not the problem: the grid was fixed in advance and the
chosen parameters were reasonable on every fold.

## Return and regime diagnostics

The dashboard's return calendar compounds net daily strategy returns within each
calendar month; the first and last months can be partial. Regime attribution
classifies a session from the **previous close** relative to its 200-session
SMA, then compares conditional daily-return statistics for the strategy and
buy-and-hold. The indicator warm-up is excluded. Because regime observations are
non-contiguous, the report avoids presenting a misleading regime CAGR.

## Limitations

One instrument at a time, long-only, daily bars. Stops are checked on closing prices only. Free Yahoo data can contain errors. A good backtest is not a forecast, and nothing here is investment advice.

Each walk-forward test window starts flat, and a position still open at the window's end is marked to market rather than closed, so the stitched series carries a small artificial cost at each boundary.

## Author

Lokesh Srirangam - [github.com/lokeshsrirangam](https://github.com/lokeshsrirangam)

## Acknowledgements

The project began from the moving-average crossover exercise in Chapter 4 of *Practical Python for Effective Algorithmic Trading* (Kuldeep Singh Rathore). The data layer, vectorised engine, metrics, tests and dashboard were rewritten for AlphaForge.
