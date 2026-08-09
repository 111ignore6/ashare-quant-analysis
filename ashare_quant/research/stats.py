from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def distribution_stats(close: pd.DataFrame) -> dict:
    """全市场日收益的分布画像：每只股票算偏度/峰度，再取横截面中位数。"""
    rets = close.pct_change().dropna(how="all")
    skews, kurts = [], []
    for col in rets.columns:
        s = rets[col].dropna()
        if len(s) < 30:
            continue
        skews.append(float(sp_stats.skew(s)))
        kurts.append(float(sp_stats.kurtosis(s)))
    return {
        "n_symbols": len(skews),
        "skewness": float(np.median(skews)) if skews else None,
        "kurtosis": float(np.median(kurts)) if kurts else None,
        "skew_mean": float(np.mean(skews)) if skews else None,
        "kurt_mean": float(np.mean(kurts)) if kurts else None,
    }


def volatility_clustering(returns: pd.Series) -> dict:
    """对指数/代表股票的日收益序列做波动聚集检验。"""
    r = returns.dropna()
    if len(r) < 100:
        return {"ljungbox_p": 1.0, "arch_p": 1.0}
    lb = acorr_ljungbox(r**2, lags=[10], return_df=True)
    arch = het_arch(r, nlags=5)
    return {"ljungbox_p": float(lb["lb_pvalue"].iloc[0]), "arch_p": float(arch[1])}
