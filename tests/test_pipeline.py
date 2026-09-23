import json

import pandas as pd
import pytest

from ashare_quant.cache import ParquetStore
from ashare_quant.config import Config
from ashare_quant.pipeline import build_panels, download_universe


def _df(dates, close):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": close, "high": [c + 0.1 for c in close], "low": [c - 0.1 for c in close],
         "close": close, "volume": [1000] * len(idx), "amount": [1e6] * len(idx)},
        index=idx,
    )


def test_download_universe_with_fake_fetcher(tmp_path):
    store = ParquetStore(tmp_path)
    cfg = Config.from_dict({"years": 1, "retry": 1})
    calls: list[str] = []

    def fake_fetcher(code, start, end, adjust):
        calls.append(code)
        return _df(["2024-01-02", "2024-01-03"], [10, 10.5])

    res = download_universe(["000001", "000002"], store, cfg, fetcher=fake_fetcher)
    assert set(calls) == {"000001", "000002"}
    assert res["ok"] == ["000001", "000002"]
    assert store.exists("000001")


def test_download_universe_skips_existing(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02"], [10]))
    cfg = Config.from_dict({"years": 1, "retry": 1})
    res = download_universe(["000001"], store, cfg, fetcher=lambda *a, **k: _df(["2024-01-03"], [11]))
    assert res["skipped"] == ["000001"]


def test_build_panels(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02", "2024-01-03"], [10, 10.5]))
    store.save("000002", _df(["2024-01-02", "2024-01-03"], [20, 19]))
    panels = build_panels(store)
    assert panels["close"].shape == (2, 2)
    assert panels["close"].columns.tolist() == ["000001", "000002"]
    assert "open" in panels and panels["open"].shape == (2, 2)


def test_build_panels_cache_hit_and_invalidate(tmp_path):
    store = ParquetStore(tmp_path)
    for i in range(10):
        code = f"{i:06d}"
        store.save(code, _df(["2024-01-02", "2024-01-03"], [10 + i, 10.5 + i]))
    p1 = build_panels(store, use_cache=True)
    assert (tmp_path / "panels" / "meta.json").exists()
    p2 = build_panels(store, use_cache=True)
    assert p1["close"].equals(p2["close"])
    # 数据更新（新增交易日）后缓存应失效并重建
    store.append("000000", _df(["2024-01-04"], [11]))
    p3 = build_panels(store, use_cache=True)
    assert p3["close"].shape == (3, 10)
    assert p3["close"].loc["2024-01-04", "000000"] == 11.0


def test_panel_cache_rejects_stale_last_date(tmp_path):
    """指纹相同但缓存面板实际最后日期滞后时必须重建。

    回归：08-12 计划任务复用了 08-11 的旧面板（指纹只含 manifest 元数据，
    回滚后 manifest 回到与缓存一致的值 → 指纹碰巧匹配），导致决策卡在
    08-11。修复：缓存 meta 记录面板最后日期，与 manifest 指数截止不符即失效。
    """
    from ashare_quant.pipeline import _load_panel_cache, _save_panel_cache

    store = ParquetStore(tmp_path)
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    panels = {
        "close": pd.DataFrame({"000001": [10.0, 10.5]}, index=idx),
        "volume": pd.DataFrame({"000001": [1000, 1000]}, index=idx),
        "open": pd.DataFrame({"000001": [10.0, 10.4]}, index=idx),
        "index_close": pd.Series([3000.0, 3010.0], index=idx),
    }
    store.update_manifest("sh000300", panels["index_close"].to_frame("close"))
    _save_panel_cache(store, "sh000300", panels)
    assert _load_panel_cache(store, "sh000300") is not None
    # 篡改 last_date：指纹未变、manifest 未变，但缓存内容实际滞后 → 必须重建
    meta_path = tmp_path / "panels" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["last_date"] = "2024-01-04"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert _load_panel_cache(store, "sh000300") is None


def test_panel_cache_invalidates_when_values_change_without_dates(tmp_path):
    """只改数值、不改日期/行数（手工数据修复）时，面板缓存必须失效。

    回归（2026-09-18 实测）：修复科创板 volume 改了 604 个 parquet 的**数值**、
    日期一行没变 → manifest 指纹与面板 last_date 都没变 → 面板缓存被判有效，
    `daily --force --retrain` 实际仍用被污染的特征（features 指纹一字未变，
    决策/重训全部建立在旧面板上）。修法：缓存 meta 记录源数据签名
    （manifest + 每个 parquet 的 mtime/size）。
    """
    store = ParquetStore(tmp_path)
    for i in range(10):
        code = f"{i:06d}"
        store.save(code, _df(["2024-01-02", "2024-01-03"], [10 + i, 10.5 + i]))
    # 必须建指数（manifest 的 index_end 是面板缓存的有效条件之一）：
    # 少了它 `_load_panel_cache` 会因 idx_end 为空**永远失效**，
    # 这样测试在旧行为下也会绿 —— 一根不会熔断的保险丝（实测踩过）。
    store.save("sh000300", _df(["2024-01-02", "2024-01-03"], [3000.0, 3010.0]))
    p1 = build_panels(store, use_cache=True)
    assert p1["close"].loc["2024-01-02", "000000"] == 10.0
    from ashare_quant.pipeline import _load_panel_cache
    assert _load_panel_cache(store, "sh000300") is not None, "前置条件：此时缓存应命中"
    # 模拟数据修复：同样的日期、同样的行数，只有数值变了
    store.save("000000", _df(["2024-01-02", "2024-01-03"], [110.0, 110.5]))
    assert _load_panel_cache(store, "sh000300") is None, \
        "数值被改写但日期未变时，面板缓存必须失效"
    p2 = build_panels(store, use_cache=True)
    assert p2["close"].loc["2024-01-02", "000000"] == 110.0


def test_download_universe_progress(capsys, tmp_path):
    store = ParquetStore(tmp_path)
    cfg = Config.from_dict({"years": 1, "retry": 1})

    def fake_fetcher(code, start, end, adjust):
        return _df(["2024-01-02"], [10])

    download_universe(["000001", "000002", "000003"], store, cfg,
                      fetcher=fake_fetcher, progress_every=1)
    captured = capsys.readouterr().out
    assert "progress 3/3" in captured


def test_download_universe_falls_back(tmp_path):
    store = ParquetStore(tmp_path)
    cfg = Config.from_dict({"years": 1, "retry": 1})

    def broken_fetcher(code, start, end, adjust):
        raise ConnectionError("primary down")

    def backup_fetcher(code, start, end, adjust):
        return _df(["2024-01-02"], [10])

    res = download_universe(["000001"], store, cfg, fetcher=broken_fetcher, fallback_fetcher=backup_fetcher)
    assert res["ok"] == ["000001"]
    assert store.exists("000001")


def _panel(closes: dict, dates):
    """{symbol: [values]} → 面板 dict（含 index_close）。"""
    idx = pd.to_datetime(dates)
    return {
        "close": pd.DataFrame(closes, index=idx),
        "volume": pd.DataFrame({k: [1.0] * len(idx) for k in closes}, index=idx),
        "open": pd.DataFrame(closes, index=idx),
        "index_close": pd.Series([3000.0] * len(idx), index=idx),
    }


def _collapsed_panel(dates, n_full: int = 60, n_last: int = 5):
    """常态 n_full 只有价，最后一日只剩 n_last 只 —— 09-16 塌缩的最小复刻。"""
    closes = {f"{i:06d}": [10.0] * len(dates) for i in range(n_full)}
    for k in sorted(closes)[n_last:]:
        closes[k][-1] = float("nan")
    return _panel(closes, dates)


def test_panel_coverage_detects_collapse(tmp_path):
    """最后一日有效股票远少于近期常态 ⇒ collapsed。

    回归：2026-09-16 面板最后一日只有 209/5360 只（3.9%），而 update_stats 自述
    healthy=true —— 放行判据用的是"自述完整性"，从没量过面板本身。
    """
    from ashare_quant.pipeline import panel_coverage

    dates = pd.date_range("2024-01-01", periods=11, freq="D")
    full = {f"{i:06d}": [10.0] * 11 for i in range(60)}
    ok = panel_coverage(_panel(full, dates))
    assert ok["collapsed"] is False and ok["has_norm"] is True
    assert ok["last_count"] == 60 and ok["ratio"] == 1.0

    bad = panel_coverage(_collapsed_panel(dates))
    assert bad["collapsed"] is True
    assert bad["last_count"] == 5 and bad["normal_count"] == 60
    assert bad["ratio"] == pytest.approx(5 / 60, abs=5e-5)  # 实现里四舍五入到 4 位

    # 历史不足 5 天时不做判定（新库/小样本不误伤）
    short = panel_coverage(_panel({f"{i:06d}": [10.0] * 3 for i in range(60)},
                                  dates[:3]))
    assert short["collapsed"] is False and short["has_norm"] is False


def test_panel_cache_not_saved_when_collapsed(tmp_path):
    """塌缩面板不得落盘：否则仪表盘（直接读 panels/close.parquet）会拿它显示一整天。"""
    from ashare_quant.pipeline import _load_panel_cache, _save_panel_cache

    store = ParquetStore(tmp_path)
    dates = pd.date_range("2024-01-01", periods=11, freq="D")
    store.save("sh000300", _df([str(d.date()) for d in dates], [3000.0] * 11))

    good = _panel({f"{i:06d}": [10.0] * 11 for i in range(60)}, dates)
    cov = _save_panel_cache(store, "sh000300", good)
    assert cov["collapsed"] is False
    assert (tmp_path / "panels" / "meta.json").exists()
    assert json.loads((tmp_path / "panels" / "meta.json").read_text(encoding="utf-8"))["coverage"]["last_count"] == 60

    # 换成"最后一日塌缩"的版本：写入侧必须拒绝
    collapsed = _collapsed_panel(dates)
    assert _save_panel_cache(store, "sh000300", collapsed)["collapsed"] is True
    # 写入被拒绝 → 磁盘上仍是上一份完好缓存（meta 的 coverage 还是 60）
    meta = json.loads((tmp_path / "panels" / "meta.json").read_text(encoding="utf-8"))
    assert meta["coverage"]["last_count"] == 60
    assert _load_panel_cache(store, "sh000300") is not None


def test_panel_cache_rejects_legacy_collapsed_cache(tmp_path):
    """磁盘上已存在的塌缩缓存（更早版本写下的）在读取时也必须被拒绝。"""
    from ashare_quant.pipeline import _load_panel_cache

    store = ParquetStore(tmp_path)
    dates = pd.date_range("2024-01-01", periods=11, freq="D")
    store.save("sh000300", _df([str(d.date()) for d in dates], [3000.0] * 11))
    cache = tmp_path / "panels"
    cache.mkdir(parents=True)
    close = _collapsed_panel(dates)["close"]
    close.to_parquet(cache / "close.parquet")
    close.to_parquet(cache / "volume.parquet")
    close.to_parquet(cache / "open.parquet")
    pd.Series([3000.0] * 11, index=dates, name="close").to_frame("close").to_parquet(
        cache / "index_close.parquet")
    # 手工伪造一份"看起来有效"的 meta（绕过写入侧守卫，模拟旧版本产物）
    from ashare_quant.pipeline import _manifest_fingerprint, _source_signature
    (cache / "meta.json").write_text(json.dumps({
        "index_symbol": "sh000300",
        "fingerprint": _manifest_fingerprint(store.read_manifest()),
        "last_date": str(dates[-1].date()),
        "source_signature": _source_signature(store),
    }), encoding="utf-8")
    assert _load_panel_cache(store, "sh000300") is None


def test_reappending_identical_index_keeps_panel_cache_hit(tmp_path):
    """`daily` 每次都 append 指数；内容没变时不得刷新 mtime，否则缓存**每天都失效**。

    回归（2026-09-18 实测）：`store.append` 无条件 `save` → 指数 parquet 的
    mtime/size 每次都变 → `_source_signature` 随之变化 → 面板缓存与特征表
    **每次运行都重建**（17:20 与 18:01 两次真实运行都打印"本次重建特征表"，
    而两次的面板内容完全一致）。修法：内容相同（`DataFrame.equals`）就不写盘。
    """
    from ashare_quant.pipeline import _load_panel_cache

    store = ParquetStore(tmp_path)
    for i in range(10):
        store.save(f"{i:06d}", _df(["2024-01-02", "2024-01-03"], [10 + i, 10.5 + i]))
    store.save("sh000300", _df(["2024-01-02", "2024-01-03"], [3000.0, 3010.0]))
    build_panels(store, use_cache=True)
    assert _load_panel_cache(store, "sh000300") is not None

    # 模拟下一次 daily 的第一阶段：指数没有新交易日，但照旧走 append
    store.append("sh000300", store.load("sh000300"))
    assert _load_panel_cache(store, "sh000300") is not None, \
        "'内容不变的重写' 不该让面板缓存失效（否则每天白重建一次）"

    # 反向保证：真的变了就必须失效（判据强度不能被这次改动削弱）
    changed = _df(["2024-01-03", "2024-01-04"], [3010.0, 3020.0])
    store.append("sh000300", changed)
    assert _load_panel_cache(store, "sh000300") is None, "指数真的多了一天时必须失效"


def test_append_same_content_does_not_touch_file(tmp_path):
    """内容相同 → 文件字节与 mtime 都不变（缓存判据依赖它）。"""
    store = ParquetStore(tmp_path)
    store.append("000001", _df(["2024-01-02"], [10.0]))
    p = tmp_path / "000001.parquet"
    before = (p.stat().st_mtime_ns, p.read_bytes())
    store.append("000001", _df(["2024-01-02"], [10.0]))      # 同日同值再 append
    after = (p.stat().st_mtime_ns, p.read_bytes())
    assert after == before
    # manifest 仍要是最新且自洽的（手工删过 manifest 的极端情形也能自愈）
    assert store.read_manifest()["000001"]["rows"] == 1
    # 数值变了 → 必须落盘
    store.append("000001", _df(["2024-01-02"], [11.0]))
    assert (p.stat().st_mtime_ns, p.read_bytes()) != before
    assert store.load("000001").iloc[0]["close"] == 11.0
