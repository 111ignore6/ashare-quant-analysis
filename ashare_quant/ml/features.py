from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 特征定义版本：改动 build_dataset 的特征列 / 窗口 / 语义时必须 +1。
# 缓存元数据里会记录它，版本不一致即视为失效（否则"代码改了、缓存没删"
# 会静默复用旧口径的特征表）。
FEATURE_CACHE_VERSION = 1


def _update_hash(h, name: str, frame) -> None:
    """把一个面板/序列的内容喂给哈希：索引、列名、形状、数值。"""
    h.update(name.encode())
    h.update("|".join(map(str, frame.index)).encode())
    if isinstance(frame, pd.DataFrame):
        h.update("|".join(map(str, frame.columns)).encode())
    arr = np.ascontiguousarray(np.asarray(frame.to_numpy(), dtype="float64"))
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())


def panel_fingerprint(close: pd.DataFrame, volume: pd.DataFrame,
                      index_close: pd.Series, horizon: int = 20,
                      require_target: bool = True,
                      target_mode: str = "raw") -> str:
    """面板**输入内容**指纹，与 `build_dataset` 的输入一一对应。

    2026-09-16 生产事故：特征缓存只校验 `as_of` 日期，面板最后一日横截面从
    204 只塌到→修回 5350 只时日期没变，于是旧缓存被判定有效，决策在塌缩的
    旧特征上重算（50 只 picks 与修复前逐字相同）。这里改成对**输入内容**做
    指纹：日期、股票列、三张表的数值、以及 horizon/require_target。

    **指纹相同 ⟹ 同一套代码重算会得到同一张特征表**（前提是 build_dataset
    未改；改了要手动 bump FEATURE_CACHE_VERSION）。反过来指纹不同就重建——
    代价只是一次 ~25s 的重算，而误用旧缓存会静默产出错误决策。

    成本实测：752×5360 的面板约 20ms/张（to_numpy+tobytes+sha256），
    相对建表 25s 可忽略。
    """
    h = hashlib.sha256()
    h.update(f"v{FEATURE_CACHE_VERSION}|horizon={int(horizon)}"
             f"|require_target={int(bool(require_target))}"
             f"|target_mode={target_mode}".encode())
    for name, frame in (("close", close), ("volume", volume),
                        ("index_close", index_close)):
        _update_hash(h, name, frame)
    return h.hexdigest()


def build_dataset(close: pd.DataFrame, volume: pd.DataFrame,
                  index_close: pd.Series, horizon: int = 20,
                  require_target: bool = True, target_mode: str = "raw"):
    """把面板拼成 长表特征集 (date, symbol)，target = 未来 horizon 日收益。

    所有特征只用当日及之前的数据（滚动窗口），横截面排名特征只用当日横截面。

    target_mode:
      "raw"    未来 horizon 日**原始**收益（历史口径）。
      "excess" 同一天**横截面去均值**后的收益。理由：策略是横截面 Top-N 等权、
               且**永远满仓**——组合收益里市场共同波动（beta）那一块无论选谁都一样，
               真正决定相对表现的是截面内的排序。用原始收益做拟合目标，模型会把
               容量花在这个当天排名用不到的共同分量上（L2 损失还会被大分量主导）。
               注意：**逐日 Spearman IC 对两者完全相同**（减去常数是保序变换），
               所以这个改动只可能体现在组合收益上，不能拿 IC 当判据。
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

    fwd = close.shift(-horizon) / close - 1
    if target_mode == "excess":
        # 横截面去均值（逐日）：只保留"选谁更好"的信号，剔除当天全市场共同涨跌
        fwd = fwd.sub(fwd.mean(axis=1), axis=0)
    target = pd.Series(fwd.to_numpy().reshape(-1), index=idx, name="target")
    mask = X.notna().all(axis=1) & target.notna()
    if not require_target:
        mask = X.notna().all(axis=1)
    return X[mask], target[mask]


def save_feature_cache(X: pd.DataFrame, y: pd.Series, path: Path, as_of,
                       fingerprint: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = X.copy()
    out["target"] = y
    out.to_parquet(path, compression="zstd")
    (path.with_suffix(".meta.json")).write_text(
        json.dumps({"as_of": str(pd.Timestamp(as_of).date()),
                    "n_rows": int(len(X)),
                    "fingerprint": fingerprint,
                    "version": FEATURE_CACHE_VERSION},
                   ensure_ascii=False), encoding="utf-8")


def load_feature_cache(path: Path, as_of, fingerprint: str | None = None):
    """读特征缓存；判据不足或面板已变时返回 None（让调用方重建）。

    校验顺序与"失效即重建"的取舍：
    1. 日期不等 → 拒绝（旧判据，保留）；
    2. `version` 与我方 FEATURE_CACHE_VERSION 不等 → 拒绝（特征定义变过）；
    3. 没给指纹，或与落盘指纹不等 → 拒绝。
       **没给指纹一律拒绝**是刻意的 fail-safe：无法验证就不复用。代价是每次
       重算 ~25s，而不是静默用错数据出决策。调用方请用
       `panel_fingerprint(...)` 或直接调用 `load_or_build_dataset(...)`。
    """
    path = Path(path)
    if not path.exists():
        return None
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if pd.Timestamp(meta["as_of"]).date() != pd.Timestamp(as_of).date():
        return None
    if meta.get("version") != FEATURE_CACHE_VERSION:
        return None
    if not fingerprint or meta.get("fingerprint") != fingerprint:
        print("提示：特征缓存缺少可校验的面板指纹（或面板已变），本次重建特征表。",
              flush=True)
        return None
    df = pd.read_parquet(path)
    if "target" not in df.columns or len(df) != meta.get("n_rows"):
        return None          # 文件被换过/写坏：元数据与实际内容自洽性检查
    y = df["target"]
    X = df.drop(columns=["target"])
    return X, y


def load_or_build_dataset(close: pd.DataFrame, volume: pd.DataFrame,
                          index_close: pd.Series, path: Path,
                          horizon: int = 20, require_target: bool = False,
                          target_mode: str = "raw"):
    """带内容校验的特征缓存：指纹一致才复用，否则重建并写回。

    cli 的每日决策走这里，保证"读缓存"与"写缓存"用同一份判据。
    target_mode 已纳入指纹，所以切换口径不会复用另一口径的缓存。
    """
    fingerprint = panel_fingerprint(close, volume, index_close, horizon,
                                    require_target, target_mode)
    cached = load_feature_cache(path, as_of=close.index.max(), fingerprint=fingerprint)
    if cached is not None:
        return cached
    X, y = build_dataset(close, volume, index_close, horizon=horizon,
                         require_target=require_target, target_mode=target_mode)
    save_feature_cache(X, y, path, as_of=close.index.max(), fingerprint=fingerprint)
    return X, y
