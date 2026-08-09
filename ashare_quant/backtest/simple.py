from __future__ import annotations

import pandas as pd


def monthly_rebalance_dates(dates) -> pd.DatetimeIndex:
    """每月第一个交易日作为调仓日。"""
    s = pd.Series(dates)
    first = s.groupby(s.dt.to_period("M")).min()
    return pd.DatetimeIndex(first.values).sort_values()


def simple_topn_returns(score: pd.DataFrame, close: pd.DataFrame,
                        rebalance_dates, top_n: int = 50) -> pd.Series:
    """每个调仓日选评分 Top N 等权，持有到下一个调仓日，返回月度组合收益。"""
    dates = [d for d in rebalance_dates if d in close.index]
    out = {}
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        s = score.loc[d].dropna()
        if len(s) < top_n:
            continue
        picks = s.nlargest(top_n).index
        ret = close.loc[nxt, picks] / close.loc[d, picks] - 1
        out[d] = float(ret.mean())
    return pd.Series(out)
