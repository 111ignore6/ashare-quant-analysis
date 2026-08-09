from __future__ import annotations

import pandas as pd


def volatility_regimes(index_close: pd.Series, n: int = 20, k: int = 3) -> pd.DataFrame:
    """用滚动波动率分位把市场分成 k 个状态（1=低波 … k=高波）。"""
    vol = index_close.pct_change().rolling(n).std()
    ranks = vol.rank(method="first")
    state = pd.qcut(ranks, k, labels=False) + 1
    return pd.DataFrame({"vol": vol, "state": state.astype("Int64")})


def state_forward_returns(index_close: pd.Series, h: int = 20, k: int = 3) -> pd.DataFrame:
    reg = volatility_regimes(index_close, k=k)
    fwd = index_close.shift(-h) / index_close - 1
    df = pd.DataFrame({"state": reg["state"], "fwd": fwd}).dropna()
    out = df.groupby("state")["fwd"].agg(mean_fwd="mean", count="count")
    return out.reset_index()
