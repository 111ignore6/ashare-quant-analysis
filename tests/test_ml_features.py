import numpy as np
import pandas as pd

import ashare_quant.ml.features as features_mod
from ashare_quant.ml.features import (
    build_dataset,
    load_feature_cache,
    load_or_build_dataset,
    panel_fingerprint,
    save_feature_cache,
)


def _market():
    rng = np.random.default_rng(12)
    idx = pd.date_range("2023-01-02", periods=160, freq="B")
    drift = np.linspace(0.0002, 0.001, 10)
    rets = rng.normal(0, 0.01, (160, 10)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(10)])
    volume = pd.DataFrame(1000, index=idx, columns=close.columns)
    index_close = close.mean(axis=1)
    return close, volume, index_close


def test_build_dataset_shape_and_target():
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)
    assert X.index.names == ["date", "symbol"]
    assert "ret_20" in X.columns and "index_state" in X.columns
    sample = X.index[100]
    d, s = sample
    assert abs(y.loc[sample] - (close.loc[close.index[close.index.get_loc(d) + 20], s] / close.loc[d, s] - 1)) < 1e-9


def test_feature_cache_roundtrip(tmp_path):
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)
    path = tmp_path / "features.parquet"
    fp = panel_fingerprint(close, volume, index_close, horizon=20)
    save_feature_cache(X, y, path, close.index.max(), fingerprint=fp)
    X2, y2 = load_feature_cache(path, close.index.max(), fingerprint=fp)
    assert X2 is not None and len(X2) == len(X)
    assert list(X2.columns) == list(X.columns)
    # 日期不同 → 拒绝（旧判据保留）
    assert load_feature_cache(path, close.index.max() - pd.Timedelta(days=1),
                              fingerprint=fp) is None


# —— 2026-09-16 生产事故回归：as_of 相同、面板内容已变，缓存必须失效 ——
def _market_with_collapse():
    """造出"塌缩面板"与"修好后面板"：最后一日横截面 2 只 vs 10 只，as_of 相同。"""
    close, volume, index_close = _market()
    last = close.index[-1]
    collapsed = close.copy()
    collapsed.loc[last, collapsed.columns[2:]] = np.nan      # 只有 2 只有当日价
    vol_collapsed = volume.copy()
    vol_collapsed.loc[last, collapsed.columns[2:]] = np.nan
    return (collapsed, vol_collapsed, close, volume, index_close, last)


def test_feature_cache_rejected_when_cross_section_collapses(tmp_path):
    """复现生产场景：as_of 不变、最后一日横截面从 2 只修回 10 只。

    旧代码只校验 as_of 日期相等 → 判定缓存有效 → 决策在塌缩的旧特征上重算
    （生产实测：修复后 features.parquet 最后一日仍 204 行、50 只 picks 与
    修复前逐字相同）。新代码必须返回 None。
    """
    collapsed, vol_collapsed, close, volume, index_close, last = _market_with_collapse()
    path = tmp_path / "features.parquet"
    X_bad, y_bad = build_dataset(collapsed, vol_collapsed, index_close,
                                 horizon=20, require_target=False)
    fp_bad = panel_fingerprint(collapsed, vol_collapsed, index_close,
                               horizon=20, require_target=False)
    save_feature_cache(X_bad, y_bad, path, last, fingerprint=fp_bad)

    # 塌缩程度可观测：缓存里最后一日只有 2 行
    n_last_cached = int((X_bad.index.get_level_values("date") == last).sum())
    assert n_last_cached == 2

    # 数据修好：同一 as_of，最后一日 10 只
    X_good, _ = build_dataset(close, volume, index_close, horizon=20, require_target=False)
    assert int((X_good.index.get_level_values("date") == last).sum()) == 10
    fp_good = panel_fingerprint(close, volume, index_close,
                                horizon=20, require_target=False)
    assert fp_good != fp_bad

    assert load_feature_cache(path, last, fingerprint=fp_good) is None   # ← 本次修复
    # 同源面板仍然命中（不能变成"永远失效"）
    hit = load_feature_cache(path, last, fingerprint=fp_bad)
    assert hit is not None and len(hit[0]) == len(X_bad)


def test_feature_cache_rejected_when_values_corrected_same_shape(tmp_path):
    """形状不变、数值被修正也必须失效（"只校验 n_rows"会漏掉的场景）。

    例：09-16 那批 bar 从腾讯 kline 估算值改成批量报价的真实值，行数一模一样。
    """
    close, volume, index_close = _market()
    path = tmp_path / "features.parquet"
    X, y = build_dataset(close, volume, index_close, horizon=20, require_target=False)
    save_feature_cache(X, y, path, close.index.max(),
                       fingerprint=panel_fingerprint(close, volume, index_close,
                                                     horizon=20, require_target=False))
    fixed = close.copy()
    fixed.iloc[-1, 0] = fixed.iloc[-1, 0] * 1.0001        # 仅一处数值修正
    assert panel_fingerprint(fixed, volume, index_close,
                             horizon=20, require_target=False) != \
        panel_fingerprint(close, volume, index_close, horizon=20, require_target=False)
    assert load_feature_cache(path, close.index.max(),
                              fingerprint=panel_fingerprint(fixed, volume, index_close,
                                                            horizon=20,
                                                            require_target=False)) is None


def test_feature_cache_without_fingerprint_is_not_reused(tmp_path):
    """没给指纹 = 无法验证 → 一律重建（fail-safe），不退回"只看日期"。"""
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)
    path = tmp_path / "features.parquet"
    save_feature_cache(X, y, path, close.index.max())        # 不带指纹
    assert load_feature_cache(path, close.index.max()) is None
    # 带指纹也读不出（落盘时就没记）
    assert load_feature_cache(path, close.index.max(),
                              fingerprint=panel_fingerprint(close, volume,
                                                            index_close, 20)) is None


def test_feature_cache_version_bump_invalidates(tmp_path, monkeypatch):
    """改了 build_dataset 的口径要 bump FEATURE_CACHE_VERSION，缓存随之失效。"""
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)
    path = tmp_path / "features.parquet"
    fp = panel_fingerprint(close, volume, index_close, horizon=20)
    save_feature_cache(X, y, path, close.index.max(), fingerprint=fp)
    assert load_feature_cache(path, close.index.max(), fingerprint=fp) is not None
    monkeypatch.setattr(features_mod, "FEATURE_CACHE_VERSION", 99)
    assert load_feature_cache(path, close.index.max(), fingerprint=fp) is None


def test_feature_cache_horizon_change_invalidates(tmp_path):
    """horizon 变了target 就变，指纹必须跟着变（缓存是 target 一起存的）。"""
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)
    path = tmp_path / "features.parquet"
    save_feature_cache(X, y, path, close.index.max(),
                       fingerprint=panel_fingerprint(close, volume, index_close, 20))
    fp10 = panel_fingerprint(close, volume, index_close, 10)
    assert fp10 != panel_fingerprint(close, volume, index_close, 20)
    assert load_feature_cache(path, close.index.max(), fingerprint=fp10) is None


def test_load_or_build_dataset_roundtrip_and_invalidation(tmp_path):
    """端到端：第一次建表并落盘，第二次命中缓存，面板一变就重建。"""
    close, volume, index_close = _market()
    path = tmp_path / "features.parquet"
    X1, y1 = load_or_build_dataset(close, volume, index_close, path)
    assert path.exists() and (tmp_path / "features.meta.json").exists()
    X2, y2 = load_or_build_dataset(close, volume, index_close, path)
    assert len(X2) == len(X1) and X2.equals(X1)
    # 同 as_of、最后一日横截面塌缩 → 必须重建（而不是复用旧表）
    collapsed = close.copy()
    collapsed.iloc[-1, 2:] = np.nan
    X3, _ = load_or_build_dataset(collapsed, volume, index_close, path)
    n_last = int((X3.index.get_level_values("date") == close.index[-1]).sum())
    assert n_last == 2, "面板已变，缓存必须失效并重建"


def _panel(n_dates: int = 100, n_syms: int = 6, seed: int = 0):
    """够长的合成面板：ret_60/vol_60 需要 ≥60 日回看。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-05", periods=n_dates)
    cols = [f"{i:06d}" for i in range(1, n_syms + 1)]
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n_dates, n_syms)), axis=0),
                      index=idx, columns=cols)
    vol = pd.DataFrame(rng.integers(1_000_000, 5_000_000, (n_dates, n_syms)).astype(float),
                       index=idx, columns=cols)
    idx_close = pd.Series(3000 * np.cumprod(1 + rng.normal(0, 0.008, n_dates)), index=idx)
    return px, vol, idx_close


def test_excess_target_is_cross_sectionally_demeaned():
    """target_mode="excess" 的逐日截面均值必须为 0；raw 则不为 0。

    为什么要这条（2026-09-16）：策略是横截面 Top-N 且永远满仓，市场共同波动
    选谁都一样。用 raw 训练会让模型把容量花在当天排名用不到的成分上。
    """
    px, vol, ic = _panel()
    _, y_raw = build_dataset(px, vol, ic, horizon=5, require_target=True,
                             target_mode="raw")
    _, y_ex = build_dataset(px, vol, ic, horizon=5, require_target=True,
                            target_mode="excess")
    assert y_ex.groupby(level="date").mean().abs().max() < 1e-12, "excess 应逐日去均值为 0"
    assert y_raw.groupby(level="date").mean().abs().mean() > 1e-6, "raw 不应被去均值"
    # 两者只在 target 上不同：同一索引、同一列
    assert y_raw.index.equals(y_ex.index)


def test_panel_fingerprint_depends_on_target_mode():
    """指纹必须随 target_mode 变化 —— 否则切换口径会复用另一口径的缓存。

    这是 2026-09-16「as_of 相同却复用塌缩旧特征」事故的同族防线：口径也是
    build_dataset 的输入，必须进指纹。
    """
    px, vol, ic = _panel()
    fr = panel_fingerprint(px, vol, ic, 5, True, "raw")
    fe = panel_fingerprint(px, vol, ic, 5, True, "excess")
    assert fr != fe, "target_mode 不同却得到同一指纹 → 缓存会串用"
    assert fr == panel_fingerprint(px, vol, ic, 5, True, "raw"), "同参数应可复现"
