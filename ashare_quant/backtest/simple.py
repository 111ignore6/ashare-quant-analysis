from __future__ import annotations

import numpy as np
import pandas as pd

# 与 backtest/engine.py 一致的默认成本（佣金 0.025% + 印花税 0.05% + 滑点 0.1%）
DEFAULT_COSTS = {"commission": 0.00025, "stamp": 0.0005, "slippage": 0.001}


def monthly_rebalance_dates(dates) -> pd.DatetimeIndex:
    """每月第一个交易日作为调仓日。"""
    s = pd.Series(dates)
    first = s.groupby(s.dt.to_period("M")).min()
    return pd.DatetimeIndex(first.values).sort_values()


def simple_topn_returns(score: pd.DataFrame, close: pd.DataFrame,
                        rebalance_dates, top_n: int = 50,
                        costs: dict | None = None) -> pd.Series:
    """每个调仓日选评分 Top N 等权，持有到下一个调仓日，返回月度组合收益。

    costs: {"commission","stamp","slippage"} → 按与上期持仓的单边换手扣费；
        None = 毛收益（**仅用于与历史口径对比，不代表可实现收益**）。

    为什么要这个参数（2026-09-16 审查）：本函数是**全部 ML 基准数字**的出处
    （benchmark.py 8 处调用、evaluate.py、screening.py），此前一律零成本。
    月频换手的成本影响小于日频，但毛收益会**系统性偏向高换手模型** ——
    拿它做模型选型，等于在奖励换手。
    """
    dates = [d for d in rebalance_dates if d in close.index]
    out = {}
    prev: set = set()
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        s = score.loc[d].dropna()
        if len(s) < top_n:
            continue
        picks = list(s.nlargest(top_n).index)
        gross = float((close.loc[nxt, picks] / close.loc[d, picks] - 1).mean())
        if not np.isfinite(gross):
            continue
        fee = 0.0
        if costs:
            c = float(costs.get("commission", 0.0))
            st = float(costs.get("stamp", 0.0))
            sl = float(costs.get("slippage", 0.0))
            turn = (1.0 - len(prev & set(picks)) / top_n) if prev else 1.0
            sold = 0.0 if not prev else turn     # 首次建仓没有卖出
            fee = sold * (c + st + sl) + turn * (c + sl)
        out[d] = gross - fee
        prev = set(picks)
    return pd.Series(out)
