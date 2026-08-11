import pandas as pd
import pytest

from ashare_quant.fetchers.tencent_fetcher import _kline, _parse_kline, to_tx_code


def test_to_tx_code():
    assert to_tx_code("600000") == "sh600000"
    assert to_tx_code("688981") == "sh688981"
    assert to_tx_code("000001") == "sz000001"
    assert to_tx_code("300750") == "sz300750"
    assert to_tx_code("920476") == "bj920476"
    assert to_tx_code("sh000300") == "sh000300"


def test_parse_kline():
    rows = [
        ["2024-01-02", "5.479", "5.449", "5.499", "5.449", "220667.000"],
        ["2024-01-03", "5.439", "5.489", "5.499", "5.439", "182037.000"],
    ]
    df = _parse_kline(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert df.index.name == "date"
    assert df.loc["2024-01-02", "close"] == 5.449
    # 手 → 股
    assert df.loc["2024-01-02", "volume"] == 22066700.0
    # 成交额 = 成交量(股) × 收盘
    assert abs(df.loc["2024-01-02", "amount"] - 22066700.0 * 5.449) < 1e-6
    assert df.loc["2024-01-02", "high"] == 5.499


def test_parse_kline_empty():
    df = _parse_kline([])
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]


def test_kline_waf_501_raises(monkeypatch):
    """腾讯 WAF 501 反爬页必须抛异常，不能被静默当成无数据（防误判停牌）。"""
    import ashare_quant.fetchers.tencent_fetcher as tf

    class FakeResp:
        status_code = 501
        text = "<html>waf.tencent.com/501page.html</html>"

        def raise_for_status(self):
            raise RuntimeError("501")

    monkeypatch.setattr(tf.requests, "get", lambda *a, **k: FakeResp())
    with pytest.raises(RuntimeError, match="风控"):
        _kline("sh600000", "2026-08-10", "2026-08-11", "qfq")
