from __future__ import annotations

import pandas as pd


def rotate_weights(model_returns: pd.DataFrame, window: int = 6, min_weight: float = 0.05,
                   change_threshold: float = 0.15) -> pd.DataFrame:
    """按滚动夏普分配权重：夏普<=0 视为 0；权重变化小于阈值时保持上次权重（防频繁切换）。"""
    rolling = model_returns.rolling(window).apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else 0.0, raw=True)
    pos = rolling.clip(lower=0)
    weights = pos.div(pos.sum(axis=1), axis=0).fillna(1 / len(model_returns.columns))
    weights = weights.clip(lower=min_weight)
    weights = weights.div(weights.sum(axis=1), axis=0)
    prev = weights.iloc[0]
    out = []
    for _, row in weights.iterrows():
        if (row - prev).abs().sum() < change_threshold:
            row = prev
        out.append(row)
        prev = row
    return pd.DataFrame(out, index=weights.index, columns=weights.columns)
