import sys

import pandas as pd
from ashare_quant.fetchers import akshare_fetcher


def test_normalize_maps_columns():
    raw = pd.DataFrame(
        {"日期": ["2024-01-02", "2024-01-03"], "开盘": [10, 10.2], "最高": [10.5, 10.6],
         "最低": [9.8, 10.0], "收盘": [10.3, 10.4], "成交量": [1000, 1200], "成交额": [1e6, 1.2e6]}
    )
    out = akshare_fetcher._normalize(raw)
    assert out.index.name == "date"
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert float(out.loc["2024-01-02", "close"]) == 10.3


def test_fetch_daily_uses_akshare(monkeypatch):
    class FakeAK:
        @staticmethod
        def stock_zh_a_daily(symbol, start_date, end_date, adjust):
            assert symbol == "sz000001"
            assert start_date == "20240101"
            return pd.DataFrame({"date": ["2024-01-02"], "open": [10], "high": [11],
                                 "low": [9], "close": [10.5], "volume": [1000], "amount": [1e6]})

    monkeypatch.setitem(sys.modules, "akshare", FakeAK())
    df = akshare_fetcher.fetch_daily("000001", "2024-01-01", "2024-01-31")
    assert len(df) == 1
    assert df.index[0] == pd.Timestamp("2024-01-02")
