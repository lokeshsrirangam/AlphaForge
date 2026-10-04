"""AlphaForge: moving-average trend backtester (Flask app).

The maths lives in alphaforge/engine.py. This file only validates input,
calls the engine and formats results for the dashboard.
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly
from flask import Flask, jsonify, render_template, request

from alphaforge import StrategyConfig, DataError, load_prices, run_backtest, synthetic_prices
from alphaforge import charts

app = Flask(__name__)


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #
def _num(data: dict, key: str, default, lo: float, hi: float, cast=float):
    raw = data.get(key, default)
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number")
    if not lo <= value <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}")
    return value


def parse_request(data: dict) -> dict:
    source = data.get("data_source", "yahoo")
    if source not in ("yahoo", "synthetic"):
        raise ValueError("data_source must be 'yahoo' or 'synthetic'")

    target_vol_pct = _num(data, "target_vol", 15, 0, 100)
    cfg = StrategyConfig(
        fast=_num(data, "short_period", 10, 2, 200, int),
        slow=_num(data, "long_period", 30, 3, 400, int),
        ma_type=data.get("ma_type", "sma"),
        stop_loss_pct=_num(data, "stop_loss", 0, 0, 50),
        target_vol=target_vol_pct / 100 if target_vol_pct > 0 else None,
        cost_bps=_num(data, "cost_bps", 10, 0, 200),
        execution_lag_days=_num(data, "execution_lag", 1, 0, 2, int),
        initial_capital=_num(data, "initial_capital", 100000, 1000, 1e9),
    )
    cfg.validate()

    ticker = str(data.get("ticker", "^NSEI")).strip()[:20]
    start = str(data.get("start_date") or "2012-01-01")
    end = str(data.get("end_date") or "") or None
    try:
        pd.to_datetime(start)
        if end:
            pd.to_datetime(end)
    except (ValueError, TypeError):
        raise ValueError("Dates must look like 2015-01-31")
    if source == "yahoo" and not ticker:
        raise ValueError("Enter a ticker symbol")

    return {
        "cfg": cfg, "source": source, "ticker": ticker, "start": start, "end": end,
        "synthetic_days": _num(data, "synthetic_days", 1500, 200, 5000, int),
        "synthetic_vol": _num(data, "synthetic_vol", 0.01, 0.001, 0.05),
        "seed": _num(data, "seed", 42, 0, 10**9, int),
    }


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def pct(x, digits=2):
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def pct_points(x, digits=2):          # value already in percent units
    return "n/a" if x is None else f"{x:.{digits}f}%"


def num(x, digits=2):
    return "n/a" if x is None else f"{x:,.{digits}f}"


def cls(x):
    return "positive" if x is not None and x >= 0 else "negative"


def build_warnings(p: dict, res) -> list[str]:
    m, b, cfg = res.metrics, res.benchmark_metrics, res.config
    out = []
    if p["source"] == "synthetic":
        out.append("Synthetic random-walk data has no edge by construction. Use it to test "
                   "the engine, not to judge the strategy.")
    if cfg.execution_lag_days == 0:
        out.append("Same-close execution assumes you can compute the signal and trade at the "
                   "same close. It is optimistic; prefer next-close.")
    if m["n_trades"] < 30:
        out.append(f"Only {m['n_trades']} trades: too few to say anything statistically meaningful.")
    if m["years"] < 5:
        out.append("Less than 5 years of data: one market regime can dominate the result.")
    if m["sharpe"] is not None and m["sharpe"] > 2:
        out.append("A Sharpe above 2 on daily bars is rare. Check for leakage, bad data or "
                   "overfitting before trusting it.")
    if m["sharpe"] is not None and m["sharpe_se"] is not None and abs(m["sharpe"]) < 2 * m["sharpe_se"]:
        out.append(f"Sharpe {m['sharpe']:.2f} is within 2 standard errors (±{m['sharpe_se']:.2f}) of "
                   "zero: it cannot be told apart from luck on this sample.")
    if b and m["sharpe"] is not None and b.get("sharpe") is not None and m["sharpe"] < b["sharpe"]:
        out.append(f"Does not beat buy & hold on risk-adjusted terms "
                   f"(Sharpe {m['sharpe']:.2f} vs {b['sharpe']:.2f}).")
    return out


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    try:
        p = parse_request(request.get_json(silent=True) or {})
        cfg = p["cfg"]
        if p["source"] == "synthetic":
            close = synthetic_prices(p["synthetic_days"], p["seed"], daily_vol=p["synthetic_vol"])
            label = "SYNTHETIC"
        else:
            close = load_prices(p["ticker"], p["start"], p["end"])
            label = p["ticker"]
            if p["ticker"].upper().endswith("-USD"):
                cfg = StrategyConfig(**{**cfg.__dict__, "periods_per_year": 365})
        res = run_backtest(close, cfg)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except DataError as exc:
        return jsonify({"error": str(exc)}), 502

    m, b, t = res.metrics, res.benchmark_metrics, res.trades

    trades = [{
        "entry_date": row.entry_date.strftime("%Y-%m-%d"),
        "entry_price": f"{row.entry_price:,.2f}",
        "exit_date": row.exit_date.strftime("%Y-%m-%d"),
        "exit_price": f"{row.exit_price:,.2f}",
        "avg_size": f"{row.avg_size * 100:.0f}%",
        "profit_loss": num(row.pnl),
        "profit_loss_percent": pct_points(row.return_pct),
        "profit_loss_class": cls(row.pnl),
        "duration": f"{row.bars} days",
        "exit_reason": row.exit_reason,
    } for row in t.itertuples()]

    rows = [("Total return", "total_return", pct), ("CAGR", "cagr", pct),
            ("Volatility", "ann_vol", pct), ("Sharpe", "sharpe", num),
            ("Sortino", "sortino", num), ("Max drawdown", "max_drawdown", pct),
            ("Calmar", "calmar", num)]
    comparison = [{"label": label_, "strategy": fmt(m.get(k)), "benchmark": fmt(b.get(k))}
                  for label_, k, fmt in rows]

    figs = {
        "price_chart": charts.price_chart(res, t),
        "equity_chart": charts.equity_chart(res),
        "drawdown_chart": charts.drawdown_chart(res),
    }

    return jsonify({
        "meta": {"label": label, "bars": int(len(res.returns)),
                 "start": res.returns.index[0].strftime("%Y-%m-%d"),
                 "end": res.returns.index[-1].strftime("%Y-%m-%d")},
        "metrics": {
            "initial_capital": num(m["initial_capital"]),
            "final_value": num(m["final_value"]),
            "total_return": pct(m["total_return"]), "total_return_class": cls(m["total_return"]),
            "annualized_return": pct(m["cagr"]), "annualized_return_class": cls(m["cagr"]),
            "sharpe": num(m["sharpe"]),
            "sharpe_se": "n/a" if m["sharpe_se"] is None else f"± {m['sharpe_se']:.2f} (1 s.e.)",
            "sortino": num(m["sortino"]), "calmar": num(m["calmar"]),
            "ann_vol": pct(m["ann_vol"]), "exposure": pct(m["exposure"], 0),
            "cost_drag": pct(m["total_cost_pct"]),
            "total_trades": m["n_trades"], "win_rate": pct(m["win_rate"]),
            "avg_win": pct_points(m["avg_win_pct"]), "avg_loss": pct_points(m["avg_loss_pct"]),
            "profit_factor": "∞" if m["profit_factor"] is None and m["n_trades"] else num(m["profit_factor"]),
            "max_drawdown": pct(m["max_drawdown"]),
        },
        "comparison": comparison,
        "warnings": build_warnings(p, res),
        "trades": trades,
        "charts": {k: json.dumps(f, cls=plotly.utils.PlotlyJSONEncoder) for k, f in figs.items()},
    })


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/docs")
def docs():
    return render_template("docs.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.context_processor
def inject_year():
    return {"current_year": datetime.now().year}


if __name__ == "__main__":
    app.run(debug=True)
