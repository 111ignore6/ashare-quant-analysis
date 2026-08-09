from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from ..backtest.metrics import drawdown_series


def equity_figure(model_returns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    equity = (1 + model_returns.fillna(0)).cumprod()
    for col in equity.columns:
        fig.add_trace(go.Scatter(x=equity.index, y=equity[col], mode="lines", name=col))
    fig.update_layout(title="模型净值曲线（模拟）", xaxis_title="日期", yaxis_title="净值")
    return fig


def drawdown_figure(model_returns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in model_returns.columns:
        dd = drawdown_series(model_returns[col])
        fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name=col))
    fig.update_layout(title="回撤曲线（模拟）", xaxis_title="日期", yaxis_title="回撤")
    return fig


def factor_heatmap(ic_summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=ic_summary.T.values, x=ic_summary.index, y=ic_summary.columns,
        colorscale="RdBu", zmid=0))
    fig.update_layout(title="因子有效性热力图（ICIR 等）", xaxis_title="因子", yaxis_title="指标")
    return fig


def build_html_report(equity: go.Figure, drawdown: go.Figure, heatmap: go.Figure,
                      log_entries: list[dict], data_through: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figs = [equity, drawdown, heatmap]
    divs = "".join(f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))
                   for i, f in enumerate(figs))
    rows = "".join(
        f"<tr><td>{e.get('date', '')}</td><td>{e.get('trigger', '')}</td>"
        f"<td>{e.get('action', '')}</td><td>{e.get('effect', '')}</td></tr>"
        for e in log_entries[-20:])
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>A股量化研究·模拟分析报告</title></head>
<body><h1>A股量化研究·模拟分析报告</h1>
<p>模拟研究，仅用于数据分析与学习，不构成投资建议。数据截止：{data_through}</p>
{divs}
<h2>调整日志</h2>
<table border="1"><tr><th>日期</th><th>触发</th><th>动作</th><th>说明</th></tr>{rows}</table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
