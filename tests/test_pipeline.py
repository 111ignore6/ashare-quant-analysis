import json

import pandas as pd
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
