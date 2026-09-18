"""关键回归测试：数据恢复、失败冷却、超时守卫、缓存文件过滤。"""

import json

import pandas as pd

from ashare_quant.cache import ParquetStore
from ashare_quant.config import Config
from ashare_quant.daily import update_daily
from ashare_quant.fetchers.registry import resolve_fallback_fetchers


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
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=failing_fetcher, fallback_fetcher=[])
    assert sorted(out["failed"]) == codes
    failed = json.loads((tmp_path / "update_failed.json").read_text(encoding="utf-8"))
    assert set(failed) == set(codes)

    # 第二次（同一天）：冷却命中，不再调用 fetcher
    calls.clear()
    out2 = update_daily(codes, store, cfg, index_fetcher=fake_index,
                        fetcher=failing_fetcher, fallback_fetcher=[])
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


def test_resolve_fallback_fetchers_multi_level():
    cfg = Config.from_dict({"data_source": "mootdx",
                            "fallback_sources": ["tencent", "akshare"]})
    fbs = resolve_fallback_fetchers(cfg)
    assert len(fbs) == 2  # tencent + akshare（不含主源）


def test_update_daily_multi_level_fallback(tmp_path, monkeypatch):
    """多级备源：主源空 → 第一备源空 → 第二备源有 → updated。"""
    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "post")
    store, codes = _setup(tmp_path, n_stocks=2, index_end="2024-01-04")
    calls = {"p": 0, "f1": 0, "f2": 0}

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def empty1(code, start, end, adjust):
        calls["p"] += 1
        return _df([], [])

    def empty2(code, start, end, adjust):
        calls["f1"] += 1
        return _df([], [])

    def good3(code, start, end, adjust):
        calls["f2"] += 1
        return _df(["2024-01-04"], [11.0])

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 2})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=empty1, fallback_fetcher=[empty2, good3])
    assert len(out["updated"]) == 2
    assert calls == {"p": 2, "f1": 2, "f2": 2}
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
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=failing_fetcher, fallback_fetcher=[])
    assert out["failed"] == codes
    assert len(calls) == 3  # 重试次数与配置一致


# —— 2026-09-16 事故回归：主源"返回了但没有目标交易日的 bar"不能被判成功 ——
def test_main_source_returning_older_bar_is_not_updated(tmp_path, monkeypatch):
    """最小复现：主源只返回 D-1 那根（新浪收盘后当日 bar 滞后数小时）。

    旧代码 `if not df.empty: return "updated"` 会把它当成功，于是
    (a) 该股没有前进一天、(b) 备源链永远不被调用。修复后必须：
    早期 bar 保留（是真实前进），但状态交给备源继续确认，最终到达 D。
    """
    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "post")
    store, codes = _setup(tmp_path, n_stocks=2, index_end="2024-01-04")
    for c in codes:                      # 股票停在 01-03，指数已到 01-04
        assert str(store.load(c).index.max().date()) == "2024-01-03"
    calls = {"primary": 0, "fallback": 0}

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def lagging_primary(code, start, end, adjust):
        calls["primary"] += 1
        return _df(["2024-01-03"], [10.6])   # 只有 D-1，没有 D

    def good_fallback(code, start, end, adjust):
        calls["fallback"] += 1
        return _df(["2024-01-04"], [11.0])   # 腾讯当日即有

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 2})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=lagging_primary, fallback_fetcher=good_fallback,
                       batch_fetcher=lambda _codes: {})
    assert calls["primary"] == 2
    assert calls["fallback"] == 2          # 关键：主源没拿到目标日必须走备源
    assert sorted(out["updated"]) == sorted(codes)
    assert out["no_data"] == []
    assert out["completeness"] == 1.0
    for c in codes:
        assert str(store.load(c).index.max().date()) == "2024-01-04"


def test_no_source_has_target_day_does_not_advance(tmp_path, monkeypatch):
    """主备源都没有当日 bar（长假前/全网延迟）：不判 updated、new_data=False。

    new_data=False 会让 cli 跳过报告与决策重算——绝不能在"指数 D + 个股 D-1"
    的塌缩面板上出决策（09-16 就是这样选出了清一色 000xxx 的"全市场 Top-50"）。
    """
    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "post")
    store, codes = _setup(tmp_path, n_stocks=3, index_end="2024-01-04")

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def lagging(code, start, end, adjust):
        return _df(["2024-01-03"], [10.6])   # 两个源都只到 D-1

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 2})
    out = update_daily(codes, store, cfg, index_fetcher=fake_index,
                       fetcher=lagging, fallback_fetcher=lagging,
                       batch_fetcher=lambda _codes: {})
    assert out["updated"] == []
    assert out["new_data"] is False
    assert out["stocks_behind"] == 3
    assert out["completeness"] == 0.0


def test_panel_cross_section_not_collapsed_when_primary_lags(tmp_path, monkeypatch):
    """端到端不变量：面板最后一日横截面不得塌缩（09-16 的真实受伤点）。

    混两条路径：一部分股票本地停在 D-1（主源空→走备源→到 D），另一部分停在
    D-2（旧代码主源返回 D-1 那根→判 updated→卡在 D-1）。旧代码下面板最后一日
    只有一半股票，决策就是在它上面选出来的。
    """
    from ashare_quant.pipeline import build_panels

    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "post")
    days = ["2024-01-02", "2024-01-03", "2024-01-04"]
    store = ParquetStore(tmp_path)
    store.save("sh000300", _df(days[:2], [3000, 3010]))
    on_time = [f"{i:06d}" for i in range(8)]        # 停在 D-1
    behind = [f"{i:06d}" for i in range(8, 16)]     # 停在 D-2
    for c in on_time:
        store.save(c, _df(days[:2], [10, 10.5]))
    for c in behind:
        store.save(c, _df(days[:1], [10]))

    def fake_index(symbol):
        return _df(days, [3000, 3010, 3020])

    def lagging_primary(code, start, end, adjust):
        # 模拟新浪：请求窗口内的 D-1 有，D 没有
        got = [d for d in days if start <= d.replace("-", "") <= end and d != "2024-01-04"]
        return _df(got, [10.5] * len(got))

    def good_fallback(code, start, end, adjust):
        got = [d for d in days if start <= d.replace("-", "") <= end]
        return _df(got, [10.5] * len(got))

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 4})
    out = update_daily([*on_time, *behind], store, cfg, index_fetcher=fake_index,
                       fetcher=lagging_primary, fallback_fetcher=good_fallback,
                       batch_fetcher=lambda _codes: {})
    assert out["completeness"] == 1.0
    close = build_panels(store, use_cache=False)["close"]
    assert str(close.index.max().date()) == "2024-01-04"
    n_last = int(close.iloc[-1].notna().sum())
    n_prev = int(close.iloc[-2].notna().sum())
    assert n_prev == 16
    assert n_last >= 0.8 * n_prev, f"面板最后一日横截面塌缩：{n_last}/{n_prev}"
    # 中间也不能留洞：停过一天的股票补的是整段
    assert not close["000008"].loc[days[1]:days[2]].isna().any()


def test_data_health_flags_stock_level_lag():
    """healthy/days_behind 必须能反映"指数到了、个股没到"（旧口径只看指数）。"""
    from ashare_quant.cli import _data_health

    today = pd.Timestamp.today().normalize()
    out = {"new_index_date": str(today.date()), "index_status": "updated",
           "stocks_total": 5360, "stocks_behind": 5150,
           "stocks_behind_expected": 371, "stocks_behind_unexpected": 4779,
           "completeness": 0.0392}
    health = _data_health(out)
    assert health["days_behind"] == 0        # 指数口径确实没落后——旧口径只看这个
    assert health["stocks_ok"] is False
    assert health["healthy"] is False
    assert health["completeness"] == 0.0392


def test_data_health_ok_when_only_suspended_stocks_behind():
    """真停牌（no_data）本来就到不了目标日，不能算故障（防假警报）。"""
    from ashare_quant.cli import _data_health

    today = pd.Timestamp.today().normalize()
    out = {"new_index_date": str(today.date()), "index_status": "updated",
           "stocks_total": 5360, "stocks_behind": 371,
           "stocks_behind_expected": 371, "stocks_behind_unexpected": 0,
           "completeness": 0.9308}
    health = _data_health(out)
    assert health["healthy"] is True
    assert health["stocks_ok"] is True


def test_data_health_without_stock_metrics_keeps_old_behaviour():
    """老调用方（不带个股口径）行为不变，避免误伤其它入口。"""
    from ashare_quant.cli import _data_health

    today = pd.Timestamp.today().normalize()
    health = _data_health({"new_index_date": str(today.date()),
                           "index_status": "updated"})
    assert health["healthy"] is True
    assert "stocks_total" not in health


def test_batch_path_not_used_intraday(tmp_path, monkeypatch):
    """盘中：指数当日 bar 被 drop_intraday_today 丢掉 → 目标日 != 报价日。

    批量报价永远给"今天"那根，而盘中不允许把今天当最新交易日，所以批量路径
    必须自动不生效（否则会把盘中价写进日线，复现 08-11 的 +0.59% 假收益）。
    """
    monkeypatch.setattr("ashare_quant.calendar.market_session", lambda: "pm")
    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "pm")
    today = pd.Timestamp.today().normalize()
    d2, d1, d0 = today - pd.Timedelta(days=2), today - pd.Timedelta(days=1), today
    store = ParquetStore(tmp_path)
    store.save("sh000300", _df([d2, d1], [3000, 3010]))   # 指数缓存停在"昨天"
    store.save("000001", _df([d2], [10.0]))
    calls = {"per_stock": 0, "batch": 0}

    def index_chain(symbol, start=None):        # 源返回含"今天"的未收盘 bar
        return _df([d2, d1, d0], [3000, 3010, 3020])

    def per_stock(code, start, end, adjust):
        calls["per_stock"] += 1
        return _df([d1], [10.5])                # 只到"昨天"（收盘确认的最后一根）

    def batch(batch_codes):
        calls["batch"] += 1
        return {c: _df([d0], [99.0]) for c in batch_codes}   # 今天那根：必须被拒

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 2})
    out = update_daily(["000001"], store, cfg, index_fetcher=index_chain,
                       fetcher=per_stock, fallback_fetcher=[], batch_fetcher=batch)
    assert calls["batch"] == 1                  # 被调用过，但一根都没被采纳
    assert calls["per_stock"] == 1              # 于是回到逐只路径
    df = store.load("000001")
    assert d0 not in df.index, "盘中不得写入今天的未收盘 bar"
    assert str(df.index.max().date()) == str(d1.date())
    assert out["updated"] == ["000001"]


# —— 出口字段完整性：任何早退分支都必须带上个股口径 ——
def test_early_return_all_up_to_date_reports_stock_health(tmp_path, monkeypatch):
    """指数无新交易日且全部股票都到位：早退分支也要报 completeness/stocks_behind。

    否则当天第二次运行（或收盘后补跑）会退回"只看指数"的旧口径——那正是
    2026-09-16 说谎的 healthy。"""
    store = ParquetStore(tmp_path)
    store.save("sh000300", _df(["2024-01-02", "2024-01-03"], [3000, 3010]))
    for code in ("000001", "000002", "000003"):
        store.save(code, _df(["2024-01-02", "2024-01-03"], [10, 10.5]))
    calls = []

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03"], [3000, 3010])

    def fake_fetcher(code, start, end, adjust):   # pragma: no cover
        calls.append(code)
        return _df(["2024-01-04"], [11.0])

    cfg = Config.from_dict({"years": 1, "retry": 1})
    out = update_daily(["000001", "000002", "000003"], store, cfg,
                       index_fetcher=fake_index, fetcher=fake_fetcher)
    assert out["up_to_date"] == "all" and out["new_data"] is False
    assert calls == []
    for key in ("stocks_total", "stocks_behind", "stocks_behind_expected",
                "stocks_behind_unexpected", "completeness"):
        assert key in out, f"早退分支缺字段 {key}"
    assert out["stocks_total"] == 3
    assert out["stocks_behind"] == 0
    assert out["stocks_behind_unexpected"] == 0
    assert out["completeness"] == 1.0


def test_cooldown_early_return_reports_stock_health(tmp_path, monkeypatch):
    """指数无新交易日 + 落后股票全在今日冷却里：字段要齐，且算作"预期内"。"""
    from ashare_quant.cli import _data_health

    monkeypatch.setattr("ashare_quant.daily.market_session", lambda: "post")
    store = ParquetStore(tmp_path)
    store.save("sh000300", _df(["2024-01-02", "2024-01-03"], [3000, 3010]))
    store.save("000001", _df(["2024-01-02"], [10.0]))       # 停在 01-02，落后一天
    store.save("000002", _df(["2024-01-02", "2024-01-03"], [10, 10.5]))
    today = str(pd.Timestamp.today().normalize().date())
    (tmp_path / "update_failed.json").write_text(
        json.dumps({"000001": today}), encoding="utf-8")     # 今日已失败 → 冷却

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03"], [3000, 3010])

    cfg = Config.from_dict({"years": 1, "retry": 1})
    out = update_daily(["000001", "000002"], store, cfg, index_fetcher=fake_index,
                       fetcher=lambda *a: _df([], []), fallback_fetcher=[])
    assert out.get("cooldown_skipped") is True
    assert out["stocks_total"] == 2
    assert out["stocks_behind"] == 1
    assert out["stocks_behind_expected"] == 1
    assert out["stocks_behind_unexpected"] == 0     # 冷却中的落后属预期内
    assert out["completeness"] == 0.5
    health = _data_health(out)
    assert health["stocks_ok"] is True              # 不能把停牌/冷却算成故障
    assert health["stocks_behind"] == 1


def test_cmd_daily_refuses_decision_when_panel_collapsed(tmp_path, monkeypatch, capsys):
    """自述 healthy 但面板横截面塌缩时，必须拒绝出报告/决策（实测 vs 自述）。

    回归：2026-09-16 的放行判据只有 update_daily **自述**的 completeness(≥0.5)，
    从没量过真正被消费的面板；自述 healthy=true 而面板最后一日只有 209/5360 只，
    当天的决策就建立在这个塌缩横截面上。现在 build_panels 会实测覆盖率
    （panel_coverage），cmd_daily 据此硬停。
    """
    from argparse import Namespace

    from ashare_quant import cli

    # 20 只股票：18 只停在 D-1，2 只到了 D → 面板最后一日 2/20
    days = [f"2024-01-{d:02d}" for d in range(1, 11)]
    store = ParquetStore(tmp_path)
    store.save("sh000300", _df([*days, "2024-01-11"], [3000.0] * 11))
    for i in range(20):
        code = f"{i:06d}"
        got = [*days, "2024-01-11"] if i < 2 else days
        store.save(code, _df(got, [10.0] * len(got)))

    monkeypatch.setattr("ashare_quant.calendar.market_session", lambda: "post")
    # 自述口径"一切正常"：这正是自述与实测背离的场景
    fake_out = {"new_index_date": "2024-01-11", "updated": ["000000", "000001"],
                "up_to_date": [], "failed": [], "no_data": [], "new_data": True,
                "stale": 0, "index_status": "updated", "index_sources": ["akshare"],
                "stocks_total": 20, "stocks_behind": 1, "stocks_behind_expected": 0,
                "stocks_behind_unexpected": 1, "completeness": 0.95}
    monkeypatch.setattr("ashare_quant.daily.update_daily", lambda *a, **k: fake_out)
    monkeypatch.setattr("ashare_quant.universe.load_universe_cached",
                        lambda *a, **k: [f"{i:06d}" for i in range(20)])

    called = []
    monkeypatch.setattr("ashare_quant.cli._build_html_report",
                        lambda *a, **k: called.append("report"))
    monkeypatch.setattr("ashare_quant.cli._save_decision",
                        lambda *a, **k: called.append("decision"))

    args = Namespace(config=str(tmp_path / "config.yaml"), data_root=str(tmp_path),
                     out_dir=str(tmp_path / "out"), model_dir=str(tmp_path / "models"),
                     sample_size=1000, retrain=False, force=False, no_decision=False)
    code = cli.cmd_daily(args)

    out = capsys.readouterr().out
    assert called == [], f"塌缩面板上不应继续跑报告/决策，实际调用了 {called}"
    assert "塌缩" in out
    assert code == cli.EXIT_DATA_FAILURE, "硬停必须给非 0 退出码，否则计划任务仍记成成功"
    stats = json.loads((tmp_path / "update_stats.json").read_text(encoding="utf-8"))
    assert stats["panel_collapsed"] is True
    assert stats["panel_coverage"]["last_count"] == 2
    assert stats["panel_coverage"]["normal_count"] == 20
    assert stats["error"] == "panel_cross_section_collapsed"
    assert stats["exit_code"] == cli.EXIT_DATA_FAILURE


def test_cmd_daily_proceeds_when_panel_healthy(tmp_path, monkeypatch):
    """反向断言：面板健康时必须照常往下走（守卫不能误伤正常运行）。"""
    from argparse import Namespace

    from ashare_quant import cli

    days = [f"2024-01-{d:02d}" for d in range(1, 11)]
    store = ParquetStore(tmp_path)
    store.save("sh000300", _df(days, [3000.0] * 10))
    for i in range(20):
        store.save(f"{i:06d}", _df(days, [10.0] * 10))

    monkeypatch.setattr("ashare_quant.calendar.market_session", lambda: "post")
    fake_out = {"new_index_date": "2024-01-10", "updated": ["000000"],
                "up_to_date": [], "failed": [], "no_data": [], "new_data": True,
                "stale": 0, "index_status": "updated", "index_sources": ["akshare"],
                "stocks_total": 20, "stocks_behind": 0, "stocks_behind_expected": 0,
                "stocks_behind_unexpected": 0, "completeness": 1.0}
    monkeypatch.setattr("ashare_quant.daily.update_daily", lambda *a, **k: fake_out)
    monkeypatch.setattr("ashare_quant.universe.load_universe_cached",
                        lambda *a, **k: [f"{i:06d}" for i in range(20)])

    called = []
    monkeypatch.setattr("ashare_quant.cli._build_html_report",
                        lambda *a, **k: called.append("report"))
    monkeypatch.setattr("ashare_quant.cli._save_decision",
                        lambda *a, **k: (called.append("decision"), ({}, {}))[1])
    monkeypatch.setattr("ashare_quant.report.daily_report.build_daily_report",
                        lambda *a, **k: called.append("daily_report"))

    args = Namespace(config=str(tmp_path / "config.yaml"), data_root=str(tmp_path),
                     out_dir=str(tmp_path / "out"), model_dir=str(tmp_path / "models"),
                     sample_size=1000, retrain=False, force=False, no_decision=False)
    code = cli.cmd_daily(args)

    assert called == ["report", "decision", "daily_report"], (
        f"面板健康时应照常跑完三段，实际 {called}")
    assert code == 0
    stats = json.loads((tmp_path / "update_stats.json").read_text(encoding="utf-8"))
    assert stats["panel_collapsed"] is False
    assert stats["panel_coverage"]["last_count"] == 20
    assert stats["exit_code"] == 0


# —— 退出码：数据侧故障必须让操作系统也看得见（2026-09-18 新增） ——
def _cmd_daily_args(tmp_path):
    from argparse import Namespace
    return Namespace(config=str(tmp_path / "config.yaml"), data_root=str(tmp_path),
                     out_dir=str(tmp_path / "out"), model_dir=str(tmp_path / "models"),
                     sample_size=1000, retrain=False, force=False, no_decision=False)


def _stub_universe(monkeypatch, n=3):
    monkeypatch.setattr("ashare_quant.universe.load_universe_cached",
                        lambda *a, **k: [f"{i:06d}" for i in range(n)])


def test_cmd_daily_exit_code_nonzero_when_all_index_sources_empty(tmp_path, monkeypatch, capsys):
    """09-17 16:05 事故回归：三个指数源全空时，退出码不能是 0。

    实测：2026-09-17 16:05 那次运行 `指数源：akshare、tencent、mootdx` 全空返回，
    数据停在 09-16、当天什么都没更新，而 `Get-ScheduledTaskInfo` 的
    LastTaskResult 仍是 **0** —— 旧代码只打印 ‼️ 就 `return None`，
    于是"计划任务一切正常"和仪表盘一样撒了同一个谎（工程约定：自述必须与实测对撞）。
    """
    from ashare_quant import cli

    monkeypatch.setattr("ashare_quant.calendar.market_session", lambda: "post")
    fake_out = {"new_index_date": "2024-01-03", "updated": [], "up_to_date": [], "failed": [],
                "no_data": [], "new_data": False, "stale": 0,
                "index_status": "all_sources_empty",
                "index_sources": ["akshare", "tencent", "mootdx"],
                "primary_index_empty": True, "error": "指数数据源全部无返回（尝试：akshare、tencent、mootdx）"}
    monkeypatch.setattr("ashare_quant.daily.update_daily", lambda *a, **k: fake_out)
    _stub_universe(monkeypatch)
    called = []
    monkeypatch.setattr("ashare_quant.cli._build_html_report", lambda *a, **k: called.append("report"))
    monkeypatch.setattr("ashare_quant.cli._save_decision", lambda *a, **k: called.append("decision"))

    code = cli.cmd_daily(_cmd_daily_args(tmp_path))

    assert called == []
    assert "全部无返回" in capsys.readouterr().out
    assert code == cli.EXIT_DATA_FAILURE
    stats = json.loads((tmp_path / "update_stats.json").read_text(encoding="utf-8"))
    assert stats["exit_code"] == cli.EXIT_DATA_FAILURE
    assert stats["error"].startswith("指数数据源全部无返回")


def test_cmd_daily_exit_code_zero_when_genuinely_up_to_date(tmp_path, monkeypatch, capsys):
    """反向断言：真的没有新交易日（例如周末补跑）不算故障，退出码必须仍是 0。

    否则计划任务每个周末都会报一次假失败，退出码就失去意义了。
    """
    from ashare_quant import cli

    monkeypatch.setattr("ashare_quant.calendar.market_session", lambda: "post")
    # 指数日期必须是"今天"，否则先撞上 days_behind 分支（另一个非 0 出口）
    today = str(pd.Timestamp.today().normalize().date())
    fake_out = {"new_index_date": today, "updated": [], "up_to_date": "all", "failed": [],
                "no_data": [], "new_data": False, "stale": 0, "index_status": "unchanged",
                "index_sources": ["akshare"], "stocks_total": 3, "stocks_behind": 0,
                "stocks_behind_expected": 0, "stocks_behind_unexpected": 0,
                "completeness": 1.0}
    monkeypatch.setattr("ashare_quant.daily.update_daily", lambda *a, **k: fake_out)
    _stub_universe(monkeypatch)

    code = cli.cmd_daily(_cmd_daily_args(tmp_path))

    assert code == 0
    assert "已是最新交易日" in capsys.readouterr().out
    stats = json.loads((tmp_path / "update_stats.json").read_text(encoding="utf-8"))
    assert stats["exit_code"] == 0


def test_cmd_daily_exit_code_nonzero_when_stocks_not_ok(tmp_path, monkeypatch, capsys):
    """指数前进但个股几乎没跟上（面板即将塌缩）：跳过重算且退出码非 0。"""
    from ashare_quant import cli

    monkeypatch.setattr("ashare_quant.calendar.market_session", lambda: "post")
    today = str(pd.Timestamp.today().normalize().date())   # 指数已到今天，故障只在个股层
    fake_out = {"new_index_date": today, "updated": ["000000"], "up_to_date": [],
                "failed": [], "no_data": [], "new_data": False, "stale": 0,
                "index_status": "updated", "index_sources": ["akshare"],
                "stocks_total": 100, "stocks_behind": 99, "stocks_behind_expected": 0,
                "stocks_behind_unexpected": 99, "completeness": 0.01}
    monkeypatch.setattr("ashare_quant.daily.update_daily", lambda *a, **k: fake_out)
    _stub_universe(monkeypatch, n=100)

    code = cli.cmd_daily(_cmd_daily_args(tmp_path))

    assert code == cli.EXIT_DATA_FAILURE
    assert "只有 1% 的股票拿到目标交易日 bar" in capsys.readouterr().out


def test_main_propagates_nonzero_exit_code(tmp_path, monkeypatch):
    """`main()` 必须把子命令的返回码变成进程退出码（计划任务靠它判成败）。"""
    from ashare_quant import cli

    monkeypatch.setattr(cli, "cmd_daily", lambda args: cli.EXIT_DATA_FAILURE)
    try:
        cli.main(["daily"])
    except SystemExit as e:
        assert e.code == cli.EXIT_DATA_FAILURE
    else:  # pragma: no cover - 旧行为（直接 args.func(args) 丢掉返回值）会走到这里
        raise AssertionError("main() 吞掉了子命令的失败退出码")

    monkeypatch.setattr(cli, "cmd_daily", lambda args: 0)
    cli.main(["daily"])            # 0 不该抛 SystemExit
