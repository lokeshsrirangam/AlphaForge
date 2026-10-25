# AlphaForge

AlphaForge is a small research app for testing trading rules on historical daily prices. It is built around a practical question: how does a strategy perform after trading costs, and how does it compare with simply holding the asset?

The app uses Python, pandas, NumPy, Flask and Plotly. It is for learning and analysis; a backtest cannot predict future returns.

## What you can do

- Load adjusted daily prices from Yahoo Finance. Downloaded data is cached locally.
- Compare a simple or exponential moving-average crossover with a Donchian close breakout.
- Set trading costs, execution timing, volatility targeting and an optional close-based stop loss.
- Review strategy and buy-and-hold performance, including CAGR, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, time in the market and trade statistics.
- Explore monthly returns and compare performance in risk-on and risk-off market regimes.
- Run walk-forward validation to choose parameters on historical training data and evaluate them on later, held-out periods.
- See warnings when a result has few trades, a short history or a Sharpe ratio that may be noise.

## Get started

Use Python 3.10 or newer. From the project folder, create an environment, install the packages and start the app:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Choose a ticker, date range and strategy, then run a backtest. Yahoo Finance data requires an internet connection. You can also choose the seeded synthetic data source to check that the engine runs without relying on market data; synthetic results do not represent real market performance.

To run the project’s tests:

```bash
python -m pytest -q
```

## How the backtest handles time

By default, a signal calculated from today’s close is executed at the next close. The position then earns returns after that execution. This delay helps prevent the strategy from using a closing price it would not have known when placing the order. Same-close execution is available for comparison, but it is an optimistic assumption.

Trading costs are charged when the position changes. The engine can also scale exposure toward a target volatility. Stop losses use closing prices, so they do not model intraday price moves or guarantee an exact exit price.

## Walk-forward validation

Walk-forward validation repeatedly chooses parameters using a training window, then evaluates that choice on the following test window. The test windows do not overlap. The default windows are 756 trading sessions for training and 252 for testing.

The parameter grid is set in advance: slow periods of 50, 100 or 200 sessions and fast periods of 5, 10, 20 or 50 sessions, keeping only pairs where the fast period is shorter. Each pair is ranked by training-period Sharpe after costs. The selected pair is then measured on the next test window, which is not used to select it.

The main walk-forward result is the combined out-of-sample return series across test windows, compared with buy and hold over those same dates. Each test window starts with no position. Any position still open at its end is marked to market, so stitching the windows can add a small boundary cost.

## Example result

The following is one historical SPY run included as an example, not a live result or a forecast. It used the defaults shown below and data through July 22, 2025.

- Training window: 756 sessions
- Test window: 252 sessions
- Trading costs: 10 basis points per position change
- Target volatility: 15% per year
- Stop loss: disabled
- Execution: next close
- Walk-forward folds: 9; combined test period: October 22, 2015 to October 25, 2024

| Measure | Strategy | Buy and hold |
| --- | ---: | ---: |
| Total return | 86.81% | 235.40% |
| CAGR | 7.19% | 14.39% |
| Sharpe ratio | 0.70 | 0.84 |
| Sharpe standard error | ±0.33 | ±0.33 |
| Maximum drawdown | -19.49% | -33.72% |

In this example, the strategy returned less than buy and hold, with a lower Sharpe ratio and a smaller maximum drawdown. The Sharpe estimate is close to its standard error, so the result is uncertain. This is a useful reminder that risk controls do not necessarily produce higher returns.

## Reading the diagnostics

Monthly returns are compounded from the strategy’s net daily returns; the first or last month in a date range may be partial. Regime labels use the previous close relative to its 200-session moving average, avoiding classification from the same day’s close. The indicator warm-up period is excluded. Regime observations are not continuous, so AlphaForge reports conditional daily-return statistics rather than a regime CAGR.

## Project structure

```text
alphaforge/       Backtest engine, market data, metrics, charts and walk-forward logic
templates/        Flask pages
static/           Styles and frontend assets
tests/            Engine and walk-forward tests
data_cache/       Locally cached market data
app.py            Flask application and API routes
```

## Known limits

AlphaForge currently tests one instrument at a time, uses daily bars and supports long-only positions. Yahoo Finance data is free and may contain gaps or errors. The simulator does not model every real-world factor, such as market impact, partial fills, taxes specific to your account or intraday execution. Results depend on the selected dates and assumptions; they are not investment advice.

## Background

The project began with the moving-average crossover exercise in Chapter 4 of *Practical Python for Effective Algorithmic Trading* by Kuldeep Singh Rathore. AlphaForge has since been extended with its own data handling, backtest engine, performance analysis, tests and web dashboard.

## Author

Lokesh Srirangam · [GitHub](https://github.com/lokeshsrirangam)
