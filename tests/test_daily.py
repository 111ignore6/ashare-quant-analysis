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
