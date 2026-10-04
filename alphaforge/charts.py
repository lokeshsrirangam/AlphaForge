"""Plotly figures. Everything is converted to plain Python lists before plotting."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objs as go


def _x(index: pd.Index) -> list[str]:
    return pd.DatetimeIndex(index).strftime("%Y-%m-%d").tolist()


def _y(series: pd.Series) -> list:
    return [None if pd.isna(v) else float(v) for v in series]


def price_chart(res, trades: pd.DataFrame) -> go.Figure:
    w = res.config.warmup
    close = res.close.iloc[w:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_x(close.index), y=_y(close), mode="lines", name="Price",
                             line=dict(color="#2E86C1", width=2)))
    fig.add_trace(go.Scatter(x=_x(close.index), y=_y(res.fast_ma.iloc[w:]), mode="lines",
                             name=f"Fast {res.config.ma_type.upper()}({res.config.fast})",
                             line=dict(color="#F39C12", width=1.5)))
    fig.add_trace(go.Scatter(x=_x(close.index), y=_y(res.slow_ma.iloc[w:]), mode="lines",
                             name=f"Slow {res.config.ma_type.upper()}({res.config.slow})",
                             line=dict(color="#C0392B", width=1.5)))
    if len(trades):
        fig.add_trace(go.Scatter(
            x=_x(pd.Index(trades["entry_date"])), y=_y(trades["entry_price"]),
            mode="markers", name="Buy (fill)",
            marker=dict(color="green", size=9, symbol="triangle-up")))
        closed = trades[~trades["exit_reason"].str.startswith("Open")]
        fig.add_trace(go.Scatter(
            x=_x(pd.Index(closed["exit_date"])), y=_y(closed["exit_price"]),
            mode="markers", name="Sell (fill)",
            marker=dict(color="red", size=9, symbol="triangle-down")))
    fig.update_layout(title="Price, moving averages and fills", template="plotly_white",
                      xaxis_title="Date", yaxis_title="Price", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def equity_chart(res) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_x(res.equity.index), y=_y(res.equity), mode="lines",
                             name="Strategy (net of costs)", line=dict(color="#2E86C1", width=2)))
    fig.add_trace(go.Scatter(x=_x(res.benchmark_equity.index), y=_y(res.benchmark_equity),
                             mode="lines", name="Buy & hold",
                             line=dict(color="#7F8C8D", width=1.5, dash="dash")))
    fig.update_layout(title="Equity curve vs buy & hold", template="plotly_white",
                      xaxis_title="Date", yaxis_title="Portfolio value", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def drawdown_chart(res) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_x(res.drawdown.index), y=_y(res.drawdown * 100), mode="lines",
                             name="Strategy", fill="tozeroy", line=dict(color="#E74C3C", width=1.5)))
    fig.add_trace(go.Scatter(x=_x(res.benchmark_drawdown.index), y=_y(res.benchmark_drawdown * 100),
                             mode="lines", name="Buy & hold",
                             line=dict(color="#7F8C8D", width=1, dash="dash")))
    fig.update_layout(title="Drawdown (%)", template="plotly_white", xaxis_title="Date",
                      yaxis_title="Drawdown (%)", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
