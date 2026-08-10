"""模拟账户：按决策持仓与行情价格计算资金、市值与盈亏。

账户视角与同类模拟盘一致：初始资金 → 按权重买入 → 每日盯市值 →
总资产/浮动盈亏/总收益率。
"""

from __future__ import annotations

import pandas as pd


def account_snapshot(decision: dict, close_panel: pd.DataFrame,
                     prices: pd.Series | None = None) -> dict | None:
    """由决策结果 + 收盘价面板计算账户快照。

    close_panel: date × symbol 收盘价矩阵（含决策日与最新价）；
    prices: 可选，最新价格覆盖（如实时行情），默认取面板最后一根。
    """
    picks = decision.get("picks") or []
    if not picks or close_panel is None or close_panel.empty:
        return None
    initial = float(decision.get("initial_capital", 100000.0))
    decision_date = pd.Timestamp(decision["date"])
    if decision_date not in close_panel.index:
        return None
    symbols = [p["symbol"] for p in picks]
    weights = pd.Series([float(p.get("weight", 0)) for p in picks], index=symbols)
    cost = close_panel.loc[decision_date, symbols].astype(float)
    latest = prices.reindex(symbols) if prices is not None \
        else close_panel.iloc[-1].reindex(symbols).astype(float)

    invest = initial * weights
    shares = invest / cost
    market_value = shares * latest
    pnl = market_value - invest
    rows = pd.DataFrame({
        "代码": symbols,
        "权重": weights.to_numpy(),
        "投入金额": invest.to_numpy(),
        "股数": shares.to_numpy(),
        "成本价": cost.to_numpy(),
        "现价": latest.to_numpy(),
        "市值": market_value.to_numpy(),
        "浮动盈亏": pnl.to_numpy(),
        "盈亏率": (pnl / invest).to_numpy(),
    })
    total_asset = float(market_value.sum())
    return {
        "rows": rows,
        "initial": initial,
        "total_asset": total_asset,
        "cash": max(0.0, initial - float(invest.sum())),
        "total_pnl": float(market_value.sum() - invest.sum()),
        "total_return": float(market_value.sum() / invest.sum() - 1)
        if invest.sum() > 0 else 0.0,
        "as_of": str(latest.name.date()) if hasattr(latest.name, "date") else str(latest.name),
    }
