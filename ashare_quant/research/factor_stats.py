from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import compute_factors, winsorize_zscore


def forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame:
    return close.shift(-h) / close - 1


def cross_sectional_ic(factor_df: pd.DataFrame, target_df: pd.DataFrame,
                       method: str = "spearman") -> pd.Series:
    """逐日计算因子值与未来收益的截面相关系数。"""
    common = factor_df.index.intersection(target_df.index)
    ic_values = {}
    for d in common:
        f = factor_df.loc[d].dropna()
        t = target_df.loc[d].dropna()
        common_cols = f.index.intersection(t.index)
        if len(common_cols) < 10:
            continue
        ic_values[d] = f[common_cols].corr(t[common_cols], method=method)
    return pd.Series(ic_values, dtype=float).sort_index()


def icir(ic: pd.Series) -> float:
    s = ic.dropna()
    return float(s.mean() / s.std() * np.sqrt(252)) if len(s) > 2 and s.std() > 0 else 0.0


def layer_returns(factor_df: pd.DataFrame, target_df: pd.DataFrame, k: int = 5) -> pd.Series:
    """按因子排名分 k 层，输出每层平均未来收益（组1=最低，组k=最高）。"""
    common = factor_df.index.intersection(target_df.index)
    layers = {d: None for d in common}
    for d in common:
        f = factor_df.loc[d].dropna()
        t = target_df.loc[d].dropna()
        common_cols = f.index.intersection(t.index)
        if len(common_cols) < k * 5:
            continue
        ranks = f[common_cols].rank(method="first")
        buckets = pd.qcut(ranks, k, labels=False) + 1
        layers[d] = t[common_cols].groupby(buckets).mean()
    stacked = pd.DataFrame(layers).T
    return stacked.mean(axis=0)


def stability_by_year(ic: pd.Series) -> pd.DataFrame:
    df = ic.to_frame("ic")
    df["year"] = df.index.year
    grp = df.groupby("year")["ic"]
    out = pd.DataFrame({
        "mean_ic": grp.mean(),
        "hit_rate": grp.apply(lambda s: (s > 0).mean()),
        "count": grp.count(),
    })
    return out


def factor_report(close: pd.DataFrame, volume: pd.DataFrame, h: int = 20,
                  k: int = 5) -> dict:
    factors = compute_factors(close, volume)
    target = forward_returns(close, h)
    summary, series = {}, {}
    for name, raw in factors.items():
        z = winsorize_zscore(raw)
        ic = cross_sectional_ic(z, target)
        layers = layer_returns(z, target, k=k)
        summary[name] = {"mean_ic": float(ic.mean()), "icir": icir(ic),
                         "layer_spread": float(layers.iloc[-1] - layers.iloc[0])}
        series[name] = ic
    return {"ic_summary": pd.DataFrame(summary).T.sort_values("icir", ascending=False),
            "ic_series": series, "layer_returns": layers}
