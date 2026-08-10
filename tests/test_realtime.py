import pandas as pd

from ashare_quant.realtime import snapshot


def test_snapshot_normalizes(monkeypatch):
    fake_quotes = {
        "600000": {"name": "浦发银行", "now": 10.0, "close": 9.5,
                   "open": 9.6, "high": 10.1, "low": 9.5, "volume": 12345},
        "600519": {"name": "贵州茅台", "now": 1500.0, "close": 1480.0,
                   "open": 1490.0, "high": 1510.0, "low": 1485.0, "volume": 678},
    }

    class FakeEQ:
        def stocks(self, codes):
            return {c: fake_quotes.get(c, {}) for c in codes}

    monkeypatch.setattr("easyquotation.use", lambda source: FakeEQ())
    df = snapshot(["600000", "600519"], source="tencent")
    assert list(df.columns) == ["代码", "名称", "现价", "涨跌幅", "今开",
                                "最高", "最低", "昨收", "成交量(手)"]
    assert len(df) == 2
    row = df[df["代码"] == "600000"].iloc[0]
    assert row["现价"] == 10.0
    assert abs(row["涨跌幅"] - (10.0 - 9.5) / 9.5) < 1e-12


def test_snapshot_skips_missing_close(monkeypatch):
    fake_quotes = {"000001": {"name": "平安银行", "now": 12.0, "close": None}}

    class FakeEQ:
        def stocks(self, codes):
            return fake_quotes

    monkeypatch.setattr("easyquotation.use", lambda source: FakeEQ())
    df = snapshot(["000001"])
    assert df.empty
