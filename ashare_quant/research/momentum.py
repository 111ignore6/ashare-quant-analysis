from __future__ import annotations

import numpy as np
import pandas as pd

from .factor_stats import cross_sectional_ic, forward_returns


def horizon_scan(close: pd.DataFrame, horizons=(5, 10, 20, 60, 120)) -> pd.DataFrame:
    """扫描各持有期的动量/反转效应：过去 h 日收益 vs 未来 h 日收益的截面 IC。"""
    rows = []
    for h in horizons:
        factor = close.pct_change(h, fill_method=None)
        target = forward_returns(close, h)
        ic = cross_sectional_ic(factor, target)
        rows.append({"horizon": h, "mean_ic": float(ic.mean()), "icir": float(_icir(ic)),
                     "fwd_mean": float(target.mean(axis=1).mean())})
    return pd.DataFrame(rows)


def _icir(ic: pd.Series) -> float:
    s = ic.dropna()
    return float(s.mean() / s.std() * np.sqrt(252)) if len(s) > 2 and s.std() > 0 else 0.0
