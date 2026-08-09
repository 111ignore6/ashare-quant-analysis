from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import compute_factors, winsorize_zscore
from .ops import ranks_axis1


def forward_returns(close: pd.DataFrame, h: int) -> pd.DataFrame:
    return close.shift(-h) / close - 1


def cross_sectional_ic(factor_df: pd.DataFrame, target_df: pd.DataFrame,
                       method: str = "spearman", min_n: int = 10) -> pd.Series:
    """逐日计算因子值与未来收益的截面相关系数（向量化，无逐日 Python 循环）。

    Spearman 等价于对每日有效截面分别 rank 后求 Pearson；
    Pearson 相关系数用逐行统计量一次性算出，数值与逐日 pandas corr 一致
    （差异 < 1e-15）。
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"unsupported method: {method}")
    valid = (factor_df.notna() & target_df.notna()).to_numpy()
    n = valid.sum(axis=1)
    f = factor_df.to_numpy(dtype=float)
    t = target_df.to_numpy(dtype=float)
    if method == "spearman":
        f, _ = ranks_axis1(np.where(valid, f, np.nan))
        t, _ = ranks_axis1(np.where(valid, t, np.nan))
    else:
        f = np.where(valid, f, np.nan)
        t = np.where(valid, t, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        fm = np.nansum(f, axis=1) / n
        tm = np.nansum(t, axis=1) / n
        cov = np.nansum(f * t, axis=1) - n * fm * tm
        varf = np.nansum(f * f, axis=1) - n * fm * fm
        vart = np.nansum(t * t, axis=1) - n * tm * tm
        corr = np.where(varf * vart > 0, cov / np.sqrt(varf * vart), np.nan)
    ok = (n >= min_n) & (varf > 0) & (vart > 0)
    return pd.Series(corr[ok], index=factor_df.index[ok]).sort_index()


def icir(ic: pd.Series) -> float:
    s = ic.dropna()
    return float(s.mean() / s.std() * np.sqrt(252)) if len(s) > 2 and s.std() > 0 else 0.0


def layer_returns(factor_df: pd.DataFrame, target_df: pd.DataFrame, k: int = 5) -> pd.Series:
    """按因子排名分 k 层，输出每层平均未来收益（组0=最低，组k-1=最高）。

    向量化：用 pandas qcut 同款分位边界公式（边界值归下桶）直接算桶号，
    避免逐日 qcut/groupby 循环；结果与旧实现逐日一致（按桶位置对齐）。
    """
    valid = (factor_df.notna() & target_df.notna()).to_numpy()
    n = valid.sum(axis=1)
    _, fr = ranks_axis1(
        np.where(valid, factor_df.to_numpy(dtype=float), np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (fr - 1) * k / (n - 1)[:, None]
    x = np.where(valid & (n > 1)[:, None], x, np.nan)
    bucket = (np.ceil(x) - 1).clip(0, k - 1)
    tgt = np.where(valid, target_df.to_numpy(dtype=float), np.nan)
    ok = n >= k * 5
    layers = {}
    for b in range(k):
        mm = (bucket == b) & ok[:, None]
        cnt = mm.sum(axis=1)
        with np.errstate(invalid="ignore"):
            layers[b] = np.nansum(np.where(mm, tgt, np.nan), axis=1) / cnt
    return pd.DataFrame(layers).mean(axis=0)


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
    tv = target.to_numpy(dtype=float)
    for name, raw in factors.items():
        z = winsorize_zscore(raw)
        valid = (z.notna() & target.notna()).to_numpy()
        n = valid.sum(axis=1)
        fv = np.where(valid, z.to_numpy(dtype=float), np.nan)
        avg_r, first_r = ranks_axis1(fv)
        tr, _ = ranks_axis1(np.where(valid, tv, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            fm = np.nansum(avg_r, axis=1) / n
            tm = np.nansum(tr, axis=1) / n
            cov = np.nansum(avg_r * tr, axis=1) - n * fm * tm
            varf = np.nansum(avg_r * avg_r, axis=1) - n * fm * fm
            vart = np.nansum(tr * tr, axis=1) - n * tm * tm
            corr = np.where(varf * vart > 0, cov / np.sqrt(varf * vart), np.nan)
        ok = (n >= 10) & (varf > 0) & (vart > 0)
        ic = pd.Series(corr[ok], index=target.index[ok]).sort_index()

        with np.errstate(invalid="ignore", divide="ignore"):
            x = (first_r - 1) * k / (n - 1)[:, None]
        x = np.where(valid & (n > 1)[:, None], x, np.nan)
        bucket = (np.ceil(x) - 1).clip(0, k - 1)
        tgt = np.where(valid, tv, np.nan)
        ok_l = n >= k * 5
        layer_cols = {}
        for b in range(k):
            mm = (bucket == b) & ok_l[:, None]
            cnt = mm.sum(axis=1)
            with np.errstate(invalid="ignore"):
                layer_cols[b] = np.nansum(np.where(mm, tgt, np.nan), axis=1) / cnt
        layers = pd.DataFrame(layer_cols).mean(axis=0)
        summary[name] = {"mean_ic": float(ic.mean()), "icir": icir(ic),
                         "layer_spread": float(layers.iloc[-1] - layers.iloc[0])}
        series[name] = ic
    return {"ic_summary": pd.DataFrame(summary).T.sort_values("icir", ascending=False),
            "ic_series": series, "layer_returns": layers}
