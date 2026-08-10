"""关键回归测试：数据恢复、失败冷却、超时守卫、缓存文件过滤。"""

import json
import time

import pandas as pd

from ashare_quant.cache import ParquetStore
from ashare_quant.config import Config
from ashare_quant.daily import update_daily


def _df(dates, close):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": close, "high": [c + 0.1 for c in close], "low": [c - 0.1 for c in close],
         "close": close, "volume": [1000] * len(idx), "amount": [1e6] * len(idx)},
        index=idx)


def _setup(tmp_path, n_stocks=3, stock_end="2024-01-03", index_end="2024-01-03"):
    store = ParquetStore(tmp_path)
    store.save("sh000300", _df(["2024-01-02", "2024-01-03", "2024-01-04"],
                               [3000, 3010, 3020]) if index_end == "2024-01-04"
               else _df(["2024-01-02", "2024-01-03"], [3000, 3010]))
    codes = []
    for i in range(n_stocks):
        code = f"{i:06d}"
        codes.append(code)
        store.save(code, _df(["2024-01-02", "2024-01-03"], [10 + i, 10.5 + i]))
    return store, codes


def test_symbols_excludes_cache_files(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02"], [10]))
    store.save("sh000300", _df(["2024-01-02"], [3000]))
    # 模拟缓存文件被放进数据目录
    (tmp_path / "features.parquet").write_bytes(b"not-a-stock")
    (tmp_path / "panels.parquet").write_bytes(b"not-a-stock")
    symbols = store.symbols()
    assert symbols == ["000001", "sh000300"]


def test_rebuild_manifest_ignores_cache_files(tmp_path):
    """rebuild_manifest 跳过 features 等非行情缓存（双层索引 parquet 会崩）。"""
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02", "2024-01-03"], [10, 10.5]))
    store.save("sh000300", _df(["2024-01-02", "2024-01-03"], [3000, 3010]))
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["000001"]],
        names=["date", "symbol"])
    pd.DataFrame({"ret_5": [1.0, 2.0]}, index=idx).to_parquet(tmp_path / "features.parquet")
    m = store.rebuild_manifest()
    assert set(m) == {"000001", "sh000300"}
    assert m["000001"]["end"] == "2024-01-03"


def test_update_daily_fills_stale_when_index_unchanged(tmp_path):
    """指数无新交易日，但股票落后（上次中断）时，应自动补齐。"""
    store, codes = _setup(tmp_path, n_stocks=3, index_end="2024-01-04")
    # 指数已在 01-04（manifest），但股票停在 01-03 → 应补齐
    calls = []

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def fake_fetcher(code, start, end, adjust):
        calls.append(code)
        assert start == "20240104"
        return _df(["2024-01-04"], [11.0])

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 3})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index, fetcher=fake_fetcher)
    assert out["new_data"] is True
    assert len(out["updated"]) == 3
    assert len(store.load(codes[0])) == 3


def test_update_daily_cooldown_skips_recent_failures(tmp_path):
    """当天失败/无数据的股票写入冷却，下次更新跳过；次日日期变化后重试。"""
    store, codes = _setup(tmp_path, n_stocks=2, index_end="2024-01-04")
    calls = []

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def failing_fetcher(code, start, end, adjust):
        calls.append(code)
        raise ConnectionError("timeout")

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 2})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index, fetcher=failing_fetcher)
    assert sorted(out["failed"]) == codes
    failed = json.loads((tmp_path / "update_failed.json").read_text(encoding="utf-8"))
    assert set(failed) == set(codes)

    # 第二次（同一天）：冷却命中，不再调用 fetcher
    calls.clear()
    out2 = update_daily(codes, store, cfg, index_fetcher=fake_index, fetcher=failing_fetcher)
    assert out2["new_data"] is False
    assert calls == []


def test_update_daily_retries_then_fails(tmp_path):
    """fetcher 持续���常时按配置重试，最终标记失败。"""
    store, codes = _setup(tmp_path, n_stocks=1, index_end="2024-01-04")
    calls = []

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def failing_fetcher(code, start, end, adjust):
        calls.append(code)
        raise ConnectionError("timeout")

    cfg = Config.from_dict({"years": 1, "retry": 3, "max_workers": 1})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index, fetcher=failing_fetcher)
    assert out["failed"] == codes
    assert len(calls) == 3  # 重试次数与配置一致
