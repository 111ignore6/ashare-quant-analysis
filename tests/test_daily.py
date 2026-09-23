import pandas as pd

from ashare_quant.cache import ParquetStore
from ashare_quant.config import Config
from ashare_quant.daily import last_trading_day, needs_update, update_daily


def _df(dates, close):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": close, "high": [c + 0.1 for c in close], "low": [c - 0.1 for c in close],
         "close": close, "volume": [1000] * len(idx), "amount": [1e6] * len(idx)},
        index=idx,
    )


def test_last_trading_day_and_needs_update(tmp_path):
    store = ParquetStore(tmp_path)
    assert last_trading_day(store) is None
    assert needs_update(store)
    store.save("sh000300", _df(["2024-01-02", "2024-01-03"], [3000, 3010]))
    assert last_trading_day(store) == pd.Timestamp("2024-01-03")
    assert not needs_update(store)


def test_update_daily_appends_only_missing(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02", "2024-01-03"], [10, 10.5]))

    def fake_index(symbol):
        assert symbol == "sh000300"
        return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])

    def fake_fetcher(code, start, end, adjust):
        assert code == "000001"
        assert start == "20240104"
        return _df(["2024-01-04"], [11.0])

    cfg = Config.from_dict({"years": 1, "retry": 1})
    out = update_daily(["000001"], store, cfg, index_fetcher=fake_index, fetcher=fake_fetcher)
    assert out["new_index_date"] == "2024-01-04"
    assert out["updated"] == ["000001"]
    assert len(store.load("000001")) == 3


def test_update_daily_fast_path_when_index_unchanged(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("sh000300", _df(["2024-01-02", "2024-01-03"], [3000, 3010]))
    store.save("000001", _df(["2024-01-02", "2024-01-03"], [10, 10.5]))
    calls = []

    def fake_index(symbol):
        return _df(["2024-01-02", "2024-01-03"], [3000, 3010])

    def fake_fetcher(code, start, end, adjust):
        calls.append(code)
        return _df(["2024-01-04"], [11.0])

    cfg = Config.from_dict({"years": 1, "retry": 1})
    out = update_daily(["000001"], store, cfg, index_fetcher=fake_index, fetcher=fake_fetcher)
    assert out["new_data"] is False
    assert out["up_to_date"] == "all"
    assert calls == []


# —— 批量报价快路径（只缺最新一根 bar 时一次 HTTP 补齐） ——
def _setup_one_new_day(tmp_path):
    """指数已到 01-04，股票停在 01-03（标准"有新交易日"增量场景）。"""
    store = ParquetStore(tmp_path)
    store.save("sh000300", _df(["2024-01-02", "2024-01-03"], [3000, 3010]))
    for code in ("000001", "000002", "000003"):
        store.save(code, _df(["2024-01-02", "2024-01-03"], [10, 10.5]))
    return store, ["000001", "000002", "000003"]


def _index_2024_01_04(symbol):
    return _df(["2024-01-02", "2024-01-03", "2024-01-04"], [3000, 3010, 3020])


def test_batch_path_fills_one_new_bar(tmp_path):
    """只缺最新一根 bar 的股票由批量报价补齐，逐只源一次都不该被调用。"""
    store, codes = _setup_one_new_day(tmp_path)

    def per_stock(code, start, end, adjust):  # pragma: no cover - 不应被调用
        raise AssertionError("批量路径应当覆盖全部股票，逐只源不该被调用")

    def batch(batch_codes):
        assert sorted(batch_codes) == codes
        return {c: _df(["2024-01-04"], [11.0]) for c in batch_codes}

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 3})
    out = update_daily(codes, store, cfg, index_fetcher=_index_2024_01_04,
                       fetcher=per_stock, fallback_fetcher=[], batch_fetcher=batch)
    assert out["updated"] == codes
    assert out["new_data"] is True
    assert out["completeness"] == 1.0
    for c in codes:
        assert len(store.load(c)) == 3
        assert str(store.load(c).index.max().date()) == "2024-01-04"
    # 批量路径也必须把 manifest 写对（下游 _symbol_end / 面板缓存指纹都读它）
    manifest = store.read_manifest()
    assert manifest["000001"] == {"start": "2024-01-02", "end": "2024-01-04", "rows": 3}


def test_batch_path_rejects_wrong_quote_date(tmp_path):
    """报价日期 != 目标交易日（盘中/源未发布）时整批放弃，退回逐只源。"""
    store, codes = _setup_one_new_day(tmp_path)
    called = []

    def per_stock(code, start, end, adjust):
        called.append(code)
        return _df(["2024-01-04"], [11.0])

    def batch(batch_codes):
        return {c: _df(["2024-01-03"], [10.9]) for c in batch_codes}  # 落后一天

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 3})
    out = update_daily(codes, store, cfg, index_fetcher=_index_2024_01_04,
                       fetcher=per_stock, fallback_fetcher=[], batch_fetcher=batch)
    assert sorted(called) == codes
    assert out["updated"] == codes
    assert all(str(store.load(c).index.max().date()) == "2024-01-04" for c in codes)


def test_batch_path_leaves_multi_day_gap_to_per_stock(tmp_path):
    """缺 2 天以上的股票不能只补最后一根（中间会留洞），必须走逐只源全区间拉。"""
    store, codes = _setup_one_new_day(tmp_path)
    store.save("000009", _df(["2024-01-02"], [9.0]))  # 停在 01-02，缺 01-03/01-04
    all_codes = [*codes, "000009"]
    per_stock_calls = []

    def per_stock(code, start, end, adjust):
        per_stock_calls.append((code, start, end))
        return _df(["2024-01-03", "2024-01-04"], [11.0, 11.5])

    def batch(batch_codes):
        assert "000009" not in batch_codes      # 不eligible
        assert sorted(batch_codes) == codes
        return {c: _df(["2024-01-04"], [11.0]) for c in batch_codes}

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 3})
    out = update_daily(all_codes, store, cfg, index_fetcher=_index_2024_01_04,
                       fetcher=per_stock, fallback_fetcher=[], batch_fetcher=batch)
    assert per_stock_calls == [("000009", "20240103", "20240104")]
    assert sorted(out["updated"]) == sorted(all_codes)
    assert len(store.load("000009")) == 3       # 无中间空洞


def test_batch_path_failure_degrades_to_per_stock(tmp_path):
    """批量取数抛异常时不能带走整批：原样退回逐只源链（等于修复前的行为）。"""
    store, codes = _setup_one_new_day(tmp_path)
    called = []

    def per_stock(code, start, end, adjust):
        called.append(code)
        return _df(["2024-01-04"], [11.0])

    def batch(batch_codes):
        raise ConnectionError("qt.gtimg.cn 不可达")

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 3})
    out = update_daily(codes, store, cfg, index_fetcher=_index_2024_01_04,
                       fetcher=per_stock, fallback_fetcher=[], batch_fetcher=batch)
    assert sorted(called) == codes
    assert out["updated"] == codes


def test_batch_path_omitted_code_goes_to_per_stock(tmp_path):
    """报价里没有的代码（停牌/退市/解析失败）继续走逐只源链判 no_data。"""
    store, codes = _setup_one_new_day(tmp_path)
    per_stock_calls = []

    def per_stock(code, start, end, adjust):
        per_stock_calls.append(code)
        return _df([], [])          # 真停牌：两源都空

    def batch(batch_codes):
        return {c: _df(["2024-01-04"], [11.0]) for c in batch_codes if c != "000002"}

    cfg = Config.from_dict({"years": 1, "retry": 1, "max_workers": 3})
    out = update_daily(codes, store, cfg, index_fetcher=_index_2024_01_04,
                       fetcher=per_stock, fallback_fetcher=[], batch_fetcher=batch)
    assert per_stock_calls == ["000002"]
    assert out["no_data"] == ["000002"]
    assert sorted(out["updated"]) == ["000001", "000003"]
    assert out["stocks_behind"] == 1
    assert out["stocks_behind_expected"] == 1     # 停牌属于预期内，不算故障


def test_batch_manifest_identical_to_per_symbol_path(tmp_path):
    """保险丝：批量路径写的 manifest 必须与 cache.py 逐只路径逐字节一致。

    `daily._update_manifest_many` 是 `ParquetStore.update_manifest` 的重复实现
    （批量路径要避免逐只 read+write 整个 manifest.json，实测 8.5ms/只 × 5360）。
    重复即隐患：manifest 指纹是整份 JSON 的 sha256（`pipeline._manifest_fingerprint`），
    格式差一个字节就会让面板缓存整个失效。这条测试锁死两条路径的等价性——
    字段、end、rows、键顺序与字节输出都要相同。
    """
    import json
    from pathlib import Path

    from ashare_quant.cache import ParquetStore as _Store
    from ashare_quant.daily import _append_merged, _update_manifest_many
    from ashare_quant.pipeline import _manifest_fingerprint

    bars = {"000001": _df(["2024-01-04"], [11.0]),
            "000002": _df(["2024-01-03", "2024-01-04"], [10.5, 11.0]),
            "000009": _df(["2024-01-04"], [9.0])}       # 000009 是 manifest 里的新条目
    old = {"000001": _df(["2024-01-02", "2024-01-03"], [10, 10.5]),
           "000002": _df(["2024-01-02"], [10.0])}

    roots = {}
    for kind in ("per_symbol", "batch"):
        root = Path(tmp_path) / kind
        store = _Store(root)
        for code, df in old.items():                    # 同一份初始状态
            store.save(code, df)
        store.save("sh000300", _df(["2024-01-02", "2024-01-03", "2024-01-04"],
                                   [3000, 3010, 3020]))
        roots[kind] = store

    # 路径 A：cache.py 原有逐只路径（save/append → update_manifest）
    for code, df in bars.items():
        roots["per_symbol"].append(code, df)
    # 路径 B：每日批量路径（先落 parquet，最后一次性写 manifest）
    frames = {code: _append_merged(roots["batch"], code, df) for code, df in bars.items()}
    _update_manifest_many(roots["batch"], frames)

    a = roots["per_symbol"].manifest_path.read_bytes()
    b = roots["batch"].manifest_path.read_bytes()
    ma, mb = json.loads(a), json.loads(b)
    assert ma == mb, "两条路径的 manifest 内容不一致"
    # 逐条核对字段（不能只靠整体相等：两边同时写错也会相等）
    for code in [*old, *bars, "sh000300"]:
        assert set(mb[code]) == {"start", "end", "rows"}, f"{code} 字段集不对"
        assert mb[code] == ma[code], f"{code} 字段值不一致"
    assert mb["000009"] == {"start": "2024-01-04", "end": "2024-01-04", "rows": 1}
    assert mb["000002"] == {"start": "2024-01-02", "end": "2024-01-04", "rows": 3}
    assert list(mb) == list(ma), "键顺序不一致（会改变 JSON 字节）"
    assert a == b, "manifest JSON 字节不一致（面板缓存指纹会失配）"
    assert _manifest_fingerprint(ma) == _manifest_fingerprint(mb)
