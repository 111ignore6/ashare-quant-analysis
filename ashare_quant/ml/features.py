from __future__ import annotations

import pandas as pd


def build_dataset(close: pd.DataFrame, volume: pd.DataFrame,
                  index_close: pd.Series, horizon: int = 20,
                  require_target: bool = True):
    """把面板拼成 长表特征集 (date, symbol)，target = 未来 horizon 日收益。

    所有特征只用当日及之前的数据（滚动窗口），横截面排名特征只用当日横截面。
    """
    ret = close.pct_change(fill_method=None)
    feats = {
        "ret_5": close.pct_change(5, fill_method=None),
        "ret_10": close.pct_change(10, fill_method=None),
        "ret_20": close.pct_change(20, fill_method=None),
        "ret_60": close.pct_change(60, fill_method=None),
        "vol_5": ret.rolling(5).std(),
        "vol_20": ret.rolling(20).std(),
        "vol_60": ret.rolling(60).std(),
        "ma_dev_5": close / close.rolling(5).mean() - 1,
        "ma_dev_20": close / close.rolling(20).mean() - 1,
        "ma_dev_60": close / close.rolling(60).mean() - 1,
        "vol_ratio": volume.rolling(5).mean() / volume.rolling(20).mean(),
    }
    frames = [f.stack(future_stack=True).rename(name) for name, f in feats.items()]
    X = pd.concat(frames, axis=1)
    X.index.names = ["date", "symbol"]
    X["cs_rank_ret20"] = close.pct_change(20, fill_method=None).rank(axis=1, pct=True).stack(future_stack=True)

    idx_ret20 = index_close.pct_change(20, fill_method=None)
    idx_vol20 = index_close.pct_change(fill_method=None).rolling(20).std()
    idx_state = pd.qcut(idx_vol20.rank(method="first"), 3, labels=False)
    idx_feats = pd.DataFrame({
        "index_ret_20": idx_ret20,
        "index_vol_20": idx_vol20,
        "index_state": idx_state.astype(float),
    })
    X = X.join(idx_feats, on="date")

    target = (close.shift(-horizon) / close - 1).stack(future_stack=True).rename("target")
    mask = X.notna().all(axis=1) & target.notna()
    if not require_target:
        mask = X.notna().all(axis=1)
    return X[mask], target[mask]
