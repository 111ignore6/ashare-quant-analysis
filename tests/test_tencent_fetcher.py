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
    df = _parse_kline(rows, "sh600000")
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert df.index.name == "date"
    assert df.loc["2024-01-02", "close"] == 5.449
    # 手 → 股
    assert df.loc["2024-01-02", "volume"] == 22066700.0
    # 成交额 = 成交量(股) × 收盘
    assert abs(df.loc["2024-01-02", "amount"] - 22066700.0 * 5.449) < 1e-6
    assert df.loc["2024-01-02", "high"] == 5.499


def test_parse_kline_empty():
    df = _parse_kline([], "sh600000")
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]


def test_parse_kline_star_market_volume_is_not_scaled():
    """科创板（688/689）腾讯给的就是"股"，不能再 ×100。

    回归：2026-09-16/17 的 688 本地数据 volume 被放大 100 倍（本地 amount 来自
    akshare 故仍正确，形成混源行）；根因是 `_parse_kline` 无条件 `* 100`。
    实测同参数：688525 原始 13,533,957 == 新浪股数，而 600000 原始 456,711 是手。
    """
    rows = [["2026-09-17", "212.01", "208.69", "215.49", "208.58", "13533957.000"]]
    star = _parse_kline(rows, "sh688525")
    assert star.loc["2026-09-17", "volume"] == 13533957.0      # 原样（股）
    main = _parse_kline(rows, "sh600000")
    assert main.loc["2026-09-17", "volume"] == 1353395700.0    # ×100（手 → 股）


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


# —— 批量报价（qt.gtimg.cn，用于每日增量补齐最新一根 bar） ——
def _quote_line(code: str, ts: str, open_: str, close: str, high: str, low: str,
                lots: str, amount: str) -> str:
    f = [""] * 50
    f[1], f[2], f[3], f[5], f[6] = "名称", code[2:], close, open_, lots
    f[30], f[33], f[34] = ts, high, low
    f[35] = f"{close}/{lots}/{amount}"
    return f'v_{code}="' + "~".join(f) + '"'


def test_parse_quotes_units_and_date():
    """报价字段 → 标准 bar：volume 手→股（×100）、amount 取真实成交额（元）。"""
    from ashare_quant.fetchers.tencent_quote import parse_quotes

    text = _quote_line("sh600000", "20260916161449", "9.170", "9.100", "9.200",
                       "9.000", "723404", "656348140")
    row = parse_quotes(text)["sh600000"]
    assert row["date"] == pd.Timestamp("2026-09-16")
    assert (row["open"], row["high"], row["low"], row["close"]) == (9.17, 9.20, 9.00, 9.10)
    assert row["volume"] == 72340400.0      # 手 → 股
    assert row["amount"] == 656348140.0     # 元（不是万元的估算值）


def test_parse_quotes_skips_suspended_and_garbage():
    """停牌（volume=0 / 价格全 0）与畸形行必须被丢弃，交给个股源链判 no_data。"""
    from ashare_quant.fetchers.tencent_quote import parse_quotes

    text = ";".join([
        _quote_line("sz301390", "20260916161430", "0.00", "37.50", "0.00", "0.00", "0", "0"),
        _quote_line("sh600825", "20260916161439", "0.00", "5.31", "0.00", "0.00", "0", "0"),
        'v_sh600000=""',
        "v_bad_line",
        _quote_line("sz000001", "20260916161427", "11.80", "11.70", "11.84", "11.57",
                    "949626", "1106652940"),
    ])
    got = parse_quotes(text)
    assert set(got) == {"sz000001"}


def test_parse_quotes_bad_timestamp_skipped():
    from ashare_quant.fetchers.tencent_quote import parse_quotes

    assert parse_quotes(_quote_line("sh600000", "bad", "9.1", "9.1", "9.2", "9.0",
                                    "100", "9100")) == {}


def test_fetch_quote_bars_survives_chunk_failure(monkeypatch):
    """分块部分失败不能带走整批：成功的块照常返回，失败的块交给个股源链。"""
    import ashare_quant.fetchers.tencent_quote as tq

    def fake_chunk(codes, tx_of):
        if codes == ["000003", "000004"]:
            raise RuntimeError("HTTP 501")
        return {c: pd.DataFrame(
            [[1.0, 1.0, 1.0, 1.0, 100.0, 100.0]], columns=tq._COLS,
            index=pd.DatetimeIndex([pd.Timestamp("2026-09-16")], name="date"))
            for c in codes}

    monkeypatch.setattr(tq, "_fetch_chunk", fake_chunk)
    codes = ["000001", "000002", "000003", "000004"]
    got = tq.fetch_quote_bars(codes, batch_size=2, workers=2)
    assert sorted(got) == ["000001", "000002"]
    assert all(df.index.max() == pd.Timestamp("2026-09-16") for df in got.values())


def test_resolve_batch_fetcher_gated_by_source_chain():
    """批量路径只在腾讯确实在源链里时启用（尊重 config 的源配置 / 测试注入假源）。"""
    from ashare_quant.daily import _resolve_batch_fetcher
    from ashare_quant.fetchers import akshare_fetcher, tencent_fetcher

    assert _resolve_batch_fetcher(akshare_fetcher.fetch_daily,
                                  [tencent_fetcher.fetch_daily]) is not None
    assert _resolve_batch_fetcher(akshare_fetcher.fetch_daily,
                                  tencent_fetcher.fetch_daily) is not None
    assert _resolve_batch_fetcher(akshare_fetcher.fetch_daily, []) is None
    fake = lambda *a, **k: None  # noqa: E731
    assert _resolve_batch_fetcher(fake, [fake]) is None
