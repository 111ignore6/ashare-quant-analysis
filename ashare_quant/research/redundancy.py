from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .factors import winsorize_zscore


def _daily_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    vals = []
    for d in a.index.intersection(b.index):
        x, y = a.loc[d].dropna(), b.loc[d].dropna()
        common = x.index.intersection(y.index)
        if len(common) >= 10:
            vals.append(x[common].corr(y[common]))
    return float(np.nanmean(vals)) if vals else 0.0


def factor_correlation(factor_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = sorted(factor_dict)
    out = pd.DataFrame(1.0, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = _daily_corr(factor_dict[a], factor_dict[b])
            out.loc[a, b] = out.loc[b, a] = c
    return out


def pca_redundancy(factor_dict: dict[str, pd.DataFrame]) -> dict:
    z = {name: winsorize_zscore(df) for name, df in factor_dict.items()}
    names = sorted(z)
    joined = pd.concat([z[n].stack().rename(n) for n in names], axis=1).dropna()
    pca = PCA(n_components=min(len(names), joined.shape[0]))
    pca.fit(joined.values)
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    n_80 = int(np.searchsorted(cum, 0.8) + 1)
    return {"first_ratio": float(ratios[0]), "cum_ratios": [float(r) for r in cum],
            "n_components_80": n_80}
