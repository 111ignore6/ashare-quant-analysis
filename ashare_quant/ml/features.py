from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    names = list(feats)
    wide = np.stack([feats[name].to_numpy() for name in names], axis=2)
    n_dates, n_symbols, n_feats = wide.shape
    idx = pd.MultiIndex.from_product(
        [close.index, close.columns], names=["date", "symbol"])
    X = pd.DataFrame(wide.reshape(n_dates * n_symbols, n_feats),
                     index=idx, columns=names)
    X["cs_rank_ret20"] = (
        close.pct_change(20, fill_method=None).rank(axis=1, pct=True)
        .to_numpy().reshape(-1))

    idx_ret20 = index_close.pct_change(20, fill_method=None)
    idx_vol20 = index_close.pct_change(fill_method=None).rolling(20).std()
    idx_state = pd.qcut(idx_vol20.rank(method="first"), 3, labels=False)
    idx_feats = pd.DataFrame({
        "index_ret_20": idx_ret20,
        "index_vol_20": idx_vol20,
        "index_state": idx_state.astype(float),
    })
    X = X.join(idx_feats, on="date")

    target = pd.Series(
        (close.shift(-horizon) / close - 1).to_numpy().reshape(-1),
        index=idx, name="target")
    mask = X.notna().all(axis=1) & target.notna()
    if not require_target:
        mask = X.notna().all(axis=1)
    return X[mask], target[mask]


def save_feature_cache(X: pd.DataFrame, y: pd.Series, path: Path, as_of) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = X.copy()
    out["target"] = y
    out.to_parquet(path)
    (path.with_suffix(".meta.json")).write_text(
        json.dumps({"as_of": str(pd.Timestamp(as_of).date()), "n_rows": int(len(X))},
                   ensure_ascii=False), encoding="utf-8")


def load_feature_cache(path: Path, as_of):
    path = Path(path)
    if not path.exists():
        return None
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if pd.Timestamp(meta["as_of"]).date() != pd.Timestamp(as_of).date():
        return None
    df = pd.read_parquet(path)
    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y
