"""自研横截面算子（纯 numpy 向量化，替代逐日 pandas 循环）。

当前提供：
- ranks_axis1：按行排名（忽略 NaN），一次排序同时产出 average / first 两种秩，
  数值与 pandas DataFrame.rank(axis=1, method=...) 一致；
- winsorize_zscore_np：按行去极值 + z-score（总体标准差），
  数值与逐行 pandas apply 一致。
"""

from __future__ import annotations

import numpy as np


def ranks_axis1(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对 (D, S) 矩阵按行排名，忽略每行 NaN。

    返回 (average_rank, first_rank)，均为 1-based，NaN 位置保持 NaN。
    一次 argsort 同时派生两种秩，避免为不同 method 重复排序。
    """
    n_rows, n_cols = a.shape
    nan = np.isnan(a)
    a_safe = np.where(nan, np.inf, a)
    order = np.argsort(a_safe, axis=1, kind="mergesort")
    sorted_a = np.take_along_axis(a_safe, order, axis=1)
    pos = np.broadcast_to(np.arange(n_cols), (n_rows, n_cols))

    first = np.empty_like(a, dtype=np.float64)
    first[np.broadcast_to(np.arange(n_rows)[:, None], a.shape), order] = pos + 1.0

    eq = sorted_a[:, 1:] == sorted_a[:, :-1]
    boundary = np.empty_like(sorted_a, dtype=bool)
    boundary[:, 0] = True
    boundary[:, 1:] = ~eq
    start_at = np.where(boundary, pos, 0)
    run_start = np.maximum.accumulate(start_at, axis=1)
    rev_marker = np.full_like(pos, n_cols)
    rev_marker[:, ::-1] = np.where(boundary, pos, n_cols)
    acc = np.minimum.accumulate(rev_marker, axis=1)
    next_start = np.empty_like(pos)
    next_start[:, ::-1] = np.concatenate(
        [np.full((n_rows, 1), n_cols), acc[:, :-1]], axis=1)
    run_len = next_start - run_start
    avg = run_start + (run_len - 1) / 2.0 + 1.0

    avg_out = np.empty_like(a, dtype=np.float64)
    avg_out[np.broadcast_to(np.arange(n_rows)[:, None], a.shape), order] = avg
    avg_out[nan] = np.nan
    first[nan] = np.nan
    return avg_out, first


def winsorize_zscore_np(a: np.ndarray, clip: float = 0.01) -> np.ndarray:
    """按行横截面：去极值（分位截断）后转 z-score（总体标准差）。

    与逐行 pandas 实现数值一致；全 NaN 行与标准差为 0 的行输出 0。
    """
    out = np.zeros_like(a)
    valid_rows = ~np.isnan(a).all(axis=1)
    if not valid_rows.any():
        return out
    sub = a[valid_rows]
    lo = np.nanquantile(sub, clip, axis=1, method="linear")
    hi = np.nanquantile(sub, 1 - clip, axis=1, method="linear")
    clipped = np.clip(sub, lo[:, None], hi[:, None])
    mu = np.nanmean(clipped, axis=1)
    sd = np.nanstd(clipped, axis=1, ddof=0)
    ok = np.isfinite(sd) & (sd > 0)
    z = np.zeros_like(clipped)
    if ok.any():
        z[ok] = (clipped[ok] - mu[ok][:, None]) / sd[ok][:, None]
    out[valid_rows] = z
    return out
