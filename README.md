# AlphaForge

A backtesting tool for a moving-average trend strategy, built to answer one question honestly: would this rule have made money **after costs**, and is the result distinguishable from luck?

Built with Python, pandas, NumPy, Flask and Plotly.

## What it does

- Loads adjusted daily prices from Yahoo Finance (cached on disk)
- SMA or EMA crossover candidates, with optional close-based stop-loss
- Volatility-targeted position sizing
- Trading costs in basis points on every change in position
- Next-close execution by default (same-close available, flagged as optimistic)
- Metrics: CAGR, volatility, Sharpe (with standard error), Sortino, max drawdown, Calmar, time in market, cost drag, trade statistics
- Strategy vs buy & hold over the same window
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

## Limitations

One instrument at a time, long-only, daily bars. Stops are checked on closing prices only. Free Yahoo data can contain errors. A good backtest is not a forecast, and nothing here is investment advice.

<!-- Results: add out-of-sample numbers here once walk-forward validation is in. -->

## Author

Lokesh Srirangam - [github.com/lokeshsrirangam](https://github.com/lokeshsrirangam)

## Acknowledgements

The project began from the moving-average crossover exercise in Chapter 4 of *Practical Python for Effective Algorithmic Trading* (Kuldeep Singh Rathore). The data layer, vectorised engine, metrics, tests and dashboard were rewritten for AlphaForge.
