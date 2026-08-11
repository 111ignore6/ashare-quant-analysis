
from ashare_quant.realtime import INDEX_CODES, index_snapshot, snapshot


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


def test_index_snapshot_returns_four(monkeypatch):
    fake_quotes = {
        "sh000001": {"name": "上证指数", "now": 3100.0, "close": 3080.0},
        "sz399001": {"name": "深证成指", "now": 10500.0, "close": 10400.0},
        "sz399006": {"name": "创业板指", "now": 2200.0, "close": 2210.0},
        "sh000300": {"name": "沪深300", "now": 3800.0, "close": 3790.0},
    }

    class FakeEQ:
        def stocks(self, codes):
            return {c: fake_quotes[c] for c in codes}

    monkeypatch.setattr("easyquotation.use", lambda source: FakeEQ())
    df = index_snapshot()
    assert list(df.columns) == ["代码", "名称", "现价", "涨跌幅"]
    assert set(df["代码"]) == set(INDEX_CODES)
    row = df[df["代码"] == "sh000001"].iloc[0]
    assert abs(row["涨跌幅"] - (3100.0 - 3080.0) / 3080.0) < 1e-12
