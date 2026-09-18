"""指数抓取链回归测试：源故障必须与"没有新交易日"区分开。

背景（2026-09-10/09-11）：mootdx 公开服务器全部无返回，`update_daily` 把
"指数取不到"静默当成"今天没有新交易日"，连续两天输出 `up_to_date="all"
失败=0`，全市场数据冻结在 09-09。
"""

import pandas as pd

from ashare_quant.cache import ParquetStore
from ashare_quant.config import Config
from ashare_quant.daily import _fetch_index, update_daily
from ashare_quant.fetchers.registry import resolve_index_fetchers


def _df(dates, close):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": close, "high": [c + 0.1 for c in close], "low": [c - 0.1 for c in close],
         "close": close, "volume": [1000] * len(idx), "amount": [1e6] * len(idx)},
        index=idx)


def _empty(symbol, start=None):
    return _df([], [])


def _dead(symbol, start=None):
    raise ConnectionError("tdx server unreachable")


def _make_store(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("sh000300", _df(["2024-01-02", "2024-01-03"], [3000, 3010]))
    store.save("000001", _df(["2024-01-02", "2024-01-03"], [10, 10.5]))
    return store


def test_resolve_index_fetchers_includes_fallback_chain():
    cfg = Config.from_dict({"data_source": "mootdx",
                            "fallback_sources": ["tencent", "akshare"]})
    names = [n for n, _ in resolve_index_fetchers(cfg)]
    # 主源在前、备源齐全；链尾固定挂着报价端点兜底源（见本文件末尾那条测试）
    assert names == ["mootdx", "tencent", "akshare", "index_snapshot"]
    assert all(callable(f) for _, f in resolve_index_fetchers(cfg))


def test_fetch_index_falls_back_and_prefers_freshest_source(monkeypatch):
    """主源挂 + 备源滞后一天 → 用能给出最新一天的那个备源，且记录生效源列表。"""
    import ashare_quant.fetchers as F

    def tx(symbol, start=None):
        return _df(["2024-01-03", "2024-01-04"], [3010, 3020])

    def sina(symbol, start=None):
        return _df(["2024-01-03"], [3010])

    monkeypatch.setattr(F, "resolve_index_fetchers",
                        lambda cfg: [("mootdx", _dead), ("tencent", tx), ("akshare", sina)])
    merged, used, primary_empty = _fetch_index(Config.from_dict({}), "sh000300",
                                               None, "2024-01-03")
    assert used == ["tencent", "akshare"]
    assert primary_empty is True
    assert merged.index.max() == pd.Timestamp("2024-01-04")
    assert float(merged.loc[pd.Timestamp("2024-01-03"), "close"]) == 3010


def test_update_daily_all_index_sources_empty_is_error_not_uptodate(tmp_path, monkeypatch):
    """全部指数源无返回：必须报错、不得声称"已最新"，也不得改动缓存。"""
    import ashare_quant.fetchers as F
    monkeypatch.setattr(F, "resolve_index_fetchers",
                        lambda cfg: [("mootdx", _dead), ("akshare", _empty)])
    store = _make_store(tmp_path)
    before = store.load("sh000300")
    calls = []

    def spy_fetcher(code, start, end, adjust):
        calls.append(code)
        return _df(["2024-01-04"], [11.0])

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 1})
    out = update_daily(["000001"], store, cfg, fetcher=spy_fetcher, fallback_fetcher=[])
    assert out["index_status"] == "all_sources_empty"
    assert out["up_to_date"] == []          # 关键：不再是 "all"
    assert out["new_data"] is False
    assert "指数数据源全部无返回" in out["error"]
    assert calls == []                       # 一天行情都不该写
    pd.testing.assert_index_equal(store.load("sh000300").index, before.index)
    # 这条早退分支同样要带个股口径（否则 update_stats 退回"只看指数"的旧口径）
    assert out["stocks_total"] == 1 and "stocks_behind" in out


def test_update_daily_survives_primary_index_failure(tmp_path, monkeypatch):
    """主源抛异常时，整批更新应靠备源继续推进到新交易日。"""
    import ashare_quant.fetchers as F
    monkeypatch.setattr(F, "resolve_index_fetchers",
                        lambda cfg: [("mootdx", _dead), ("tencent",
                                     lambda s, start=None: _df(["2024-01-04"], [3020]))])
    store = _make_store(tmp_path)
    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 1})
    out = update_daily(["000001"], store, cfg,
                       fetcher=lambda c, s, e, a: _df(["2024-01-04"], [11.0]),
                       fallback_fetcher=[])
    assert out["new_index_date"] == "2024-01-04"
    assert out["updated"] == ["000001"]
    assert out["primary_index_empty"] is True
    assert len(store.load("sh000300")) == 3


def test_index_chain_always_ends_with_quote_snapshot():
    """链尾必须固定挂着"报价端点"兜底源（2026-09-18 新增）。

    实测（09-18 16:07）：能当日给指数的历史 K 线端点只有腾讯，而它返回 HTTP 501（WAF）；
    新浪指数天然滞后一个交易日、通达信不可用 —— 没有这一级，`_fetch_index` 判"全部空返回"
    或停在 D-1，整批更新就永远推不动（09-17、09-18 两个 16:05 任务连续空转即此）。
    缺了它这条测试必须红。
    """
    from ashare_quant.fetchers import index_snapshot

    cfg = Config.from_dict({"data_source": "akshare", "fallback_sources": ["tencent"]})
    chain = resolve_index_fetchers(cfg)
    names = [n for n, _ in chain]
    assert names[-1] == "index_snapshot", f"指数链尾应为报价兜底源，实际 {names}"
    assert chain[-1][1] is index_snapshot.fetch_index_daily
    assert names[0] == "akshare" and "tencent" in names
