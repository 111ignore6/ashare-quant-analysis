"""指数"报价端点兜底源"的回归（2026-09-18 新增，`ashare_quant/fetchers/index_snapshot.py`）。

钉住三件事：
1. **解析**：两家报价端点的字段位置与单位（成交量是"手"，×100 才等于本地指数库的"股"）；
   两个真实响应文本是 2026-09-18 16:07 实测抓下的（同一时刻腾讯 K 线端点是 HTTP 501）。
2. **降级**：腾讯报价失败要自动换新浪，两家都失败返回空表（不抛、不阻断主链）。
3. **盘中安全（最容易出错的一条）**：盘中拿到的当日快照**必须被 `drop_intraday_today`
   丢掉**，绝不能把未收盘价当收盘价写进指数日线（08-11 假收益那类事故）。
"""

from __future__ import annotations

import pandas as pd
import pytest

from ashare_quant.fetchers import index_snapshot as isn

# —— 2026-09-18 16:07 实测抓下的真实响应（腾讯 K 线同时刻为 HTTP 501） ——
TENCENT_REAL = (
    'v_sh000300="1~沪深300~000300~4507.39~4460.16~4492.32~189924166~0~0~0.00~0~0.00~0~0.00~'
    '0~0.00~0~0.00~0~0.00~0~0.00~0~0.00~0~0.00~0~0.00~0~~20260918160714~47.23~1.06~4523.09~'
    '4480.59~4507.39/189924166/537699359227~189924166~53769936~0.57~13.39~~4523.09~4480.59~'
    '0.95~512563.96~540615.81~0.00~-1~-1~1.14~0~4";'
)
SINA_REAL = (
    'var hq_str_sh000300="沪深300,4492.3233,4460.1557,4507.3926,4523.0915,4480.5856,0,0,'
    '189924166,537699359227,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'
    '2026-09-18,15:35:44,00,";'
)

_VOL_HAND = 189_924_166          # 两家的成交量字段（手）
_AMOUNT = 537_699_359_227.0      # 两家的成交额字段（元）


def _tencent_body(date: pd.Timestamp, close=4507.39, open_=4492.32, high=4523.09, low=4480.59):
    ts = date.strftime("%Y%m%d") + "160714"
    return (f'v_sh000300="1~沪深300~000300~{close}~4460.16~{open_}~{_VOL_HAND}~0~0~0.00~'
            f'0~0.00~0~0.00~0~0.00~0~~{ts}~47.23~1.06~{high}~{low}~'
            f'{close}/{_VOL_HAND}/{int(_AMOUNT)}~{_VOL_HAND}~53769936";')


def _sina_body(date: pd.Timestamp, close=4507.3926, open_=4492.3233, high=4523.0915, low=4480.5856):
    return (f'var hq_str_sh000300="沪深300,{open_},4460.1557,{close},{high},{low},0,0,'
            f'{_VOL_HAND},{int(_AMOUNT)},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'
            f'{date:%Y-%m-%d},15:35:44,00,";')


def test_parse_tencent_snapshot_fields_and_unit():
    row = isn.parse_tencent(TENCENT_REAL)
    assert row is not None
    assert row["date"] == pd.Timestamp("2026-09-18")
    assert (row["open"], row["high"], row["low"], row["close"]) == (4492.32, 4523.09,
                                                                    4480.59, 4507.39)
    # 报价端点给的是"手"，本地指数库是"股"（09-17 本地 1.700256e10 == ak 的 17002564200）
    assert row["volume"] == _VOL_HAND * isn.HAND_TO_SHARE
    assert row["amount"] == _AMOUNT


def test_parse_sina_snapshot_matches_tencent():
    """两家的日期/收盘/量额一致（实测两者字段值逐位相同）→ 互为交叉验证。"""
    t, s = isn.parse_tencent(TENCENT_REAL), isn.parse_sina(SINA_REAL)
    assert t["date"] == s["date"] == pd.Timestamp("2026-09-18")
    assert abs(t["close"] - s["close"]) < 0.01
    assert t["volume"] == s["volume"] == _VOL_HAND * isn.HAND_TO_SHARE
    assert t["amount"] == s["amount"] == _AMOUNT


@pytest.mark.parametrize("bad", ["", "garbage", 'var x="";', 'v_sh000300="1~沪深300";'])
def test_parse_returns_none_on_garbage(bad):
    assert isn.parse_tencent(bad) is None
    assert isn.parse_sina(bad) is None


def test_high_low_fallback_when_endpoint_omits_them():
    """端点偶尔缺高/低（填 0）→ 用开收盘兜底，绝不让 high < close 这种脏值进日线。"""
    body = ('v_sh000300="1~沪深300~000300~4507.39~4460.16~4492.32~189924166~0~0~0.00~'
            '0~0.00~0~~20260918160714~47.23~1.06~0~0~4507.39/189924166/537699359227";')
    row = isn.parse_tencent(body)
    assert row["high"] == 4507.39 and row["low"] == 4492.32


def test_fetch_falls_back_to_sina(monkeypatch):
    def fake_get(url):
        if "qt.gtimg" in url:                       # 腾讯报价失败（如 WAF）
            raise RuntimeError("HTTP 501")
        return SINA_REAL
    monkeypatch.setattr(isn, "_get", fake_get)
    df = isn.fetch_index_daily("sh000300")
    assert len(df) == 1 and df.index[0] == pd.Timestamp("2026-09-18")
    assert df.iloc[0]["close"] == pytest.approx(4507.3926)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]


def test_fetch_returns_empty_when_both_sources_fail(monkeypatch):
    monkeypatch.setattr(isn, "_get", lambda url: (_ for _ in ()).throw(RuntimeError("down")))
    df = isn.fetch_index_daily("sh000300")
    assert df.empty and list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]


def test_fetch_daily_refuses_to_be_a_stock_source():
    """本模块没有个股日线；被误配成 data_source 时要响亮失败，而不是静默返回空。"""
    with pytest.raises(RuntimeError, match="指数兜底"):
        isn.fetch_daily("600000", "20260101", "20260131", "qfq")


def test_snapshot_row_is_dropped_intraday(monkeypatch):
    """盘中（含午间）不认当日快照 → `_call_index` 返回 None；收盘后才认。

    日期用"今天"构造（不能硬编码，否则测试第二天就过期）。
    """
    from ashare_quant import calendar as cal
    from ashare_quant.daily import _call_index

    today = pd.Timestamp.today().normalize()
    monkeypatch.setattr(isn, "_get", lambda url: _tencent_body(today))
    monkeypatch.setattr(cal, "market_session", lambda: "pm")
    assert _call_index(isn.fetch_index_daily, "sh000300", None) is None, \
        "盘中不得把未收盘的当日快照写进指数日线"

    monkeypatch.setattr(cal, "market_session", lambda: "post")
    df = _call_index(isn.fetch_index_daily, "sh000300", None)
    assert df is not None and len(df) == 1 and df.index[0] == today
