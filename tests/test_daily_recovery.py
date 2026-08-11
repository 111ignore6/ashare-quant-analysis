"""关键回归测试：数据恢复、失败冷却、超时守卫、缓存文件过滤。"""

import json

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
    assert out2.get("cooldown_skipped") is True
    assert out2["up_to_date"] == "all"
    assert calls == []


def test_update_daily_fallback_on_empty_response(tmp_path, monkeypatch):
    """主源返回空（限流/盘中无当日 bar 的误判）时，应交给备源确认补齐。"""
    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "post")
    store, codes = _setup(tmp_path, n_stocks=2, index_end="2024-01-04")
    calls = {"primary": 0, "fallback": 0}

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def empty_fetcher(code, start, end, adjust):
        calls["primary"] += 1
        return _df([], [])

    def good_fetcher(code, start, end, adjust):
        calls["fallback"] += 1
        return _df(["2024-01-04"], [11.0])

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 2})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=empty_fetcher, fallback_fetcher=good_fetcher)
    assert len(out["updated"]) == 2
    assert calls["primary"] == 2 and calls["fallback"] == 2
    assert len(store.load(codes[0])) == 3


def test_update_daily_skips_fallback_intraday(tmp_path, monkeypatch):
    """盘中时段：主源空时不再等备源（新浪当日日线未生成），避免整批拖慢。"""
    store, codes = _setup(tmp_path, n_stocks=2, index_end="2024-01-04")
    calls = {"primary": 0, "fallback": 0}
    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "lunch")

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def empty_fetcher(code, start, end, adjust):
        calls["primary"] += 1
        return _df([], [])

    def good_fetcher(code, start, end, adjust):
        calls["fallback"] += 1
        return _df(["2024-01-04"], [11.0])

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 2})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=empty_fetcher, fallback_fetcher=good_fetcher)
    assert out["no_data"] == codes
    assert calls["fallback"] == 0  # 盘中不等待备源


def test_update_daily_uses_fallback_after_hours(tmp_path, monkeypatch):
    """收盘后：主源空时走备源确认（限流/真停牌）。"""
    store, codes = _setup(tmp_path, n_stocks=2, index_end="2024-01-04")
    calls = {"primary": 0, "fallback": 0}
    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "post")

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def empty_fetcher(code, start, end, adjust):
        calls["primary"] += 1
        return _df([], [])

    def good_fetcher(code, start, end, adjust):
        calls["fallback"] += 1
        return _df(["2024-01-04"], [11.0])

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 2})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=empty_fetcher, fallback_fetcher=good_fetcher)
    assert len(out["updated"]) == 2
    assert calls["fallback"] == 2


def test_update_daily_no_data_when_both_sources_empty(tmp_path):
    """主备源都返回空 → no_data（真停牌），不算 failed。"""
    store, codes = _setup(tmp_path, n_stocks=1, index_end="2024-01-04")

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def empty_fetcher(code, start, end, adjust):
        return _df([], [])

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 1})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=empty_fetcher, fallback_fetcher=empty_fetcher)
    assert out["no_data"] == codes
    assert out["failed"] == []


def test_update_daily_backoff_on_many_failures(tmp_path, monkeypatch):
    """连续大量异常失败（非 no_data）时批量层暂停 60 秒，避免加重行情源风控。"""
    store, codes = _setup(tmp_path, n_stocks=30, index_end="2024-01-04")
    sleeps = []
    monkeypatch.setattr("ashare_quant.daily.time.sleep", lambda s: sleeps.append(s))

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def failing_fetcher(code, start, end, adjust):
        raise ConnectionError("timeout")

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 30})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=failing_fetcher, fallback_fetcher=failing_fetcher)
    assert len(out["failed"]) == 30
    assert 60 in sleeps


def test_update_daily_no_data_does_not_backoff(tmp_path, monkeypatch):
    """no_data（停牌/盘中未生成）是正常结果，不触发 60 秒退避。"""
    store, codes = _setup(tmp_path, n_stocks=30, index_end="2024-01-04")
    sleeps = []
    monkeypatch.setattr("ashare_quant.daily.time.sleep", lambda s: sleeps.append(s))

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def empty_fetcher(code, start, end, adjust):
        return _df([], [])

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 30})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=empty_fetcher, fallback_fetcher=empty_fetcher)
    assert len(out["no_data"]) == 30
    assert 60 not in sleeps


def test_update_daily_no_data_cooldown_only_after_hours(tmp_path, monkeypatch):
    """盘中 no_data 不写冷却（收盘后要重试）；收盘后 no_data 才冷却（真停牌）。"""
    store, codes = _setup(tmp_path, n_stocks=2, index_end="2024-01-04")

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def empty_fetcher(code, start, end, adjust):
        return _df([], [])

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 2})

    # 盘中：no_data 不写冷却
    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "lunch")
    out1 = update_daily(codes, store, cfg, index_fetcher=fake_index,
                        fetcher=empty_fetcher, fallback_fetcher=empty_fetcher)
    assert out1["no_data"] == codes
    assert not (tmp_path / "update_failed.json").exists()

    # 收盘后：no_data 写冷却
    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "post")
    out2 = update_daily(codes, store, cfg, index_fetcher=fake_index,
                        fetcher=empty_fetcher, fallback_fetcher=empty_fetcher)
    assert out2["no_data"] == codes
    failed = json.loads((tmp_path / "update_failed.json").read_text(encoding="utf-8"))
    assert set(failed) == set(codes)


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
