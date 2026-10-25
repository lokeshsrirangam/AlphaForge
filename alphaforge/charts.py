"""Plotly figures. Everything is converted to plain Python lists before plotting."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objs as go

CHART_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "#111c2d",
    "plot_bgcolor": "#111c2d",
    "font": {"family": "Manrope, Inter, sans-serif", "color": "#b7c5d8", "size": 11},
    "margin": {"l": 54, "r": 20, "t": 48, "b": 42},
    "hoverlabel": {"bgcolor": "#1a2a40", "bordercolor": "#30445f", "font": {"color": "#edf3fb"}},
    "xaxis": {"gridcolor": "#213147", "linecolor": "#304158", "zerolinecolor": "#304158"},
    "yaxis": {"gridcolor": "#213147", "linecolor": "#304158", "zerolinecolor": "#304158"},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
}


def _x(index: pd.Index) -> list[str]:
    return pd.DatetimeIndex(index).strftime("%Y-%m-%d").tolist()


def _y(series: pd.Series) -> list:
    return [None if pd.isna(v) else float(v) for v in series]


def price_chart(res, trades: pd.DataFrame) -> go.Figure:
    w = res.config.warmup
    close = res.close.iloc[w:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_x(close.index), y=_y(close), mode="lines", name="Price",
                             line=dict(color="#66A4FF", width=2)))
    if res.config.strategy_type == "donchian_breakout":
        entry = res.close.rolling(res.config.slow, min_periods=res.config.slow).max().shift(1)
        exit_ = res.close.rolling(res.config.fast, min_periods=res.config.fast).min().shift(1)
        fig.add_trace(go.Scatter(x=_x(close.index), y=_y(entry.iloc[w:]), mode="lines",
                                 name=f"Entry channel ({res.config.slow})",
                                 line=dict(color="#31C48D", width=1.5)))
        fig.add_trace(go.Scatter(x=_x(close.index), y=_y(exit_.iloc[w:]), mode="lines",
                                 name=f"Exit channel ({res.config.fast})",
                                 line=dict(color="#A78BFA", width=1.5)))
        title = "Close-based Donchian channels and fills"
    else:
        fig.add_trace(go.Scatter(x=_x(close.index), y=_y(res.fast_ma.iloc[w:]), mode="lines",
                                 name=f"Fast {res.config.ma_type.upper()}({res.config.fast})",
                                 line=dict(color="#31C48D", width=1.5)))
        fig.add_trace(go.Scatter(x=_x(close.index), y=_y(res.slow_ma.iloc[w:]), mode="lines",
                                 name=f"Slow {res.config.ma_type.upper()}({res.config.slow})",
                                 line=dict(color="#A78BFA", width=1.5)))
        title = "Price, moving averages and fills"
    if len(trades):
        fig.add_trace(go.Scatter(
            x=_x(pd.Index(trades["entry_date"])), y=_y(trades["entry_price"]),
            mode="markers", name="Buy (fill)",
            marker=dict(color="#31C48D", size=9, symbol="triangle-up")))
        closed = trades[~trades["exit_reason"].str.startswith("Open")]
        fig.add_trace(go.Scatter(
            x=_x(pd.Index(closed["exit_date"])), y=_y(closed["exit_price"]),
            mode="markers", name="Sell (fill)",
            marker=dict(color="#FB7185", size=9, symbol="triangle-down")))
    fig.update_layout(**CHART_LAYOUT, title=title,
                      xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
    return fig


def equity_chart(res) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_x(res.equity.index), y=_y(res.equity), mode="lines",
                             name="Strategy (net of costs)", line=dict(color="#66A4FF", width=2)))
    fig.add_trace(go.Scatter(x=_x(res.benchmark_equity.index), y=_y(res.benchmark_equity),
                             mode="lines", name="Buy & hold",
                             line=dict(color="#99A8BC", width=1.5, dash="dash")))
    fig.update_layout(**CHART_LAYOUT, title="Equity curve vs buy & hold",
                      xaxis_title="Date", yaxis_title="Portfolio value", hovermode="x unified")
    return fig


def drawdown_chart(res) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_x(res.drawdown.index), y=_y(res.drawdown * 100), mode="lines",
                             name="Strategy", fill="tozeroy", fillcolor="rgba(251,113,133,0.16)", line=dict(color="#FB7185", width=1.5)))
    fig.add_trace(go.Scatter(x=_x(res.benchmark_drawdown.index), y=_y(res.benchmark_drawdown * 100),
                             mode="lines", name="Buy & hold",
                             line=dict(color="#99A8BC", width=1, dash="dash")))
    fig.update_layout(**CHART_LAYOUT, title="Drawdown (%)", xaxis_title="Date",
                      yaxis_title="Drawdown (%)", hovermode="x unified")
    return fig


def monthly_return_chart(res) -> go.Figure:
    """Calendar view of compounded monthly strategy returns."""
    months = res.returns.groupby(res.returns.index.to_period("M")).apply(
        lambda values: (1.0 + values).prod() - 1.0
    )
    years = sorted({period.year for period in months.index})
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    z = [[None for _ in range(12)] for _ in years]
    labels = [["" for _ in range(12)] for _ in years]
    for period, value in months.items():
        row = years.index(period.year)
        col = period.month - 1
        z[row][col] = float(value * 100.0)
        labels[row][col] = f"{value:+.1%}"

    fig = go.Figure(go.Heatmap(
        x=month_names, y=[str(year) for year in years], z=z, text=labels,
        texttemplate="%{text}",
        colorscale=[[0.0, "#7f3544"], [0.5, "#18283c"], [1.0, "#16805f"]],
        zmid=0, colorbar=dict(title="Return %", ticksuffix="%", outlinewidth=0),
        hovertemplate="%{y} %{x}<br>Return: %{z:.2f}%<extra></extra>",
        xgap=4, ygap=4,
    ))
    layout = dict(CHART_LAYOUT)
    layout["yaxis"] = {**CHART_LAYOUT["yaxis"], "autorange": "reversed"}
    layout["margin"] = {"l": 54, "r": 65, "t": 48, "b": 42}
    fig.update_layout(**layout, title="Monthly strategy returns",
                      xaxis_title="Month", yaxis_title="Year")
    return fig
