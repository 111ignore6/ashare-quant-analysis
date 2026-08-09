from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factors(close: pd.DataFrame, volume: pd.DataFrame,
                    n_short: int = 5, n_long: int = 20, n_vol: int = 20) -> dict[str, pd.DataFrame]:
    """每天对每只股票计算技术因子（未标准化）。"""
    ret = close.pct_change(fill_method=None)
    return {
        "momentum": close.pct_change(n_long, fill_method=None),
        "reversal": -close.pct_change(n_short, fill_method=None),
        "volatility": ret.rolling(n_vol).std(),
        "ma_deviation": close / close.rolling(n_long).mean() - 1,
        "volume_ratio": volume.rolling(n_short).mean() / volume.rolling(n_long).mean(),
    }


def winsorize_zscore(df: pd.DataFrame, clip: float = 0.01) -> pd.DataFrame:
    """逐日横截面：先去极值（分位截断），再转 z-score（总体标准差，保证每行均值0、标准差1）。"""
    def _row(row: pd.Series) -> pd.Series:
        lo, hi = row.quantile(clip), row.quantile(1 - clip)
        clipped = row.clip(lo, hi)
        mu, sd = clipped.mean(), clipped.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=row.index)
        return (clipped - mu) / sd

    return df.apply(_row, axis=1)
