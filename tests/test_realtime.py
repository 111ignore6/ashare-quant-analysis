"""实时行情快照解析与降级测试。

回归背景（2026-09-18 实测）：`easyquotation` 0.7.7 把腾讯/新浪接口都硬编码成
明文 http，而两家已对 http 返回 400 空响应；该库又把空响应当"字段不足"静默
丢弃 → 返回空 dict 且不抛异常，仪表盘只能提示"可能非交易时段或接口限流"。
因此这里钉住三件事：走 https、主源失败降级备源、全失败必须显式抛错。
"""

import pytest

from ashare_quant.realtime import INDEX_CODES, RealtimeError, _prefix, index_snapshot, snapshot


def _tencent_line(prefixed: str, name: str, now: float, prev: float,
                  open_=None, high=None, low=None, vol_hand=1234.0) -> str:
    """按腾讯真实格式造一行（88 字段，字段 3=现价 4=昨收 6=成交量(手) 33=高 34=低）。"""
    f = [""] * 88
    f[0] = f'v_{prefixed}="1'
    f[1] = name
    f[2] = prefixed[-6:]
    f[3] = str(now)
    f[4] = str(prev)
    f[5] = str(prev if open_ is None else open_)
    f[6] = str(vol_hand)
    f[33] = str(now if high is None else high)
    f[34] = str(now if low is None else low)
    return "~".join(f)


def _tencent_payload(mapping: dict[str, tuple]) -> str:
    return ";".join(_tencent_line(k, *v) for k, v in mapping.items()) + ";"


def _sina_payload(mapping: dict[str, tuple]) -> str:
    """新浪格式：名称,今开,昨收,现价,最高,最低,买一,卖一,成交量(股),成交额。"""
    lines = []
    for code, (name, open_, prev, now, high, low, vol_share) in mapping.items():
        lines.append(f'var hq_str_{code}="{name},{open_},{prev},{now},{high},{low},'
                     f'{now},{now},{vol_share},1000.0,2026-09-18,13:01:00,00";')
    return "\n".join(lines)


def test_star_market_quote_volume_normalized_to_hand(monkeypatch):
    """科创板报价成交量腾讯给的是"股"，本表统一成"手"（÷100）；其余板块本就是手。

    实测（2026-09-18）：688525 腾讯 13,854,702 == 新浪股数；600000 腾讯 312,748
    == 新浪股数/100。若不做区分，688 的成交量会虚高 100 倍。
    """
    payload = _tencent_payload({
        "sh688525": ("佰维存储", 218.07, 208.27, 215.0, 218.6, 211.23, 13854702.0),
        "sh600000": ("浦发银行", 9.06, 9.06, 9.05, 9.15, 9.00, 312748.0),
    })
    monkeypatch.setattr("ashare_quant.realtime._get", lambda url: payload)
    df = snapshot(["688525", "600000"]).set_index("代码")
    assert df.loc["688525", "成交量(手)"] == 138547.02   # 股 → 手
    assert df.loc["600000", "成交量(手)"] == 312748.0    # 本就是手，不再变


def test_tencent_url_is_https():
    """根因回归：接口必须走 https —— 明文 http 已被两家返回 400。"""
    from ashare_quant import realtime

    assert realtime.TENCENT_URL.startswith("https://")
    assert realtime.SINA_URL.startswith("https://")


def test_snapshot_parses_tencent(monkeypatch):
    payload = _tencent_payload({
        "sh600000": ("浦发银行", 10.0, 9.5, 9.6, 10.1, 9.5, 304647.0),
        "sz000001": ("平安银行", 12.0, 12.0, 11.9, 12.1, 11.8, 5000.0),
    })
    monkeypatch.setattr("ashare_quant.realtime._get", lambda url: payload)

    df = snapshot(["600000", "000001"], source="tencent")
    assert list(df.columns) == ["代码", "名称", "现价", "涨跌幅", "今开", "最高",
                                "最低", "昨收", "成交量(手)"]
    assert list(df["代码"]) == ["600000", "000001"]  # 6 位码，按请求顺序
    row = df[df["代码"] == "600000"].iloc[0]
    assert row["现价"] == 10.0
    assert row["昨收"] == 9.5
    assert abs(row["涨跌幅"] - (10.0 - 9.5) / 9.5) < 1e-12
    assert row["最高"] == 10.1 and row["最低"] == 9.5
    # 成交量列单位必须是"手"（仪表盘拼当日 K 线时会 ×100 换算成股；
    # 早期实现塞的是"股"，会让成交量柱放大 100 倍）
    assert row["成交量(手)"] == 304647.0


def test_bj_codes_use_bj_prefix(monkeypatch):
    """北交所 920xxx 必须走 bj 前缀（用 sh 前缀拿不到数据）。"""
    seen = {}

    def fake_get(url):
        seen["url"] = url
        return _tencent_payload({"bj920100": ("北交所股", 20.0, 19.0)})

    monkeypatch.setattr("ashare_quant.realtime._get", fake_get)
    df = snapshot(["920100"])
    assert "bj920100" in seen["url"]
    assert df.iloc[0]["代码"] == "920100"
    assert _prefix("920100") == "bj920100"
    assert _prefix("600000") == "sh600000"
    assert _prefix("000001") == "sz000001"
    assert _prefix("sh000300") == "sh000300"  # 已带前缀原样返回


def test_snapshot_falls_back_to_sina(monkeypatch):
    """主源失败（如 400/超时）时必须自动降级到备源，而不是返回空表。"""
    sina = _sina_payload({"sh600000": ("浦发银行", 9.05, 9.06, 9.06, 9.15, 9.00,
                                       30384761.0)})

    def fake_get(url):
        if "gtimg" in url:
            raise RealtimeError("HTTP 400（https://qt.gtimg.cn/q）")
        return sina

    monkeypatch.setattr("ashare_quant.realtime._get", fake_get)
    df = snapshot(["600000"])
    assert len(df) == 1
    assert df.iloc[0]["名称"] == "浦发银行"
    assert df.iloc[0]["现价"] == 9.06
    # 新浪成交量单位是股 → 统一换算成手
    assert abs(df.iloc[0]["成交量(手)"] - 303847.61) < 1e-6


def test_snapshot_raises_when_all_sources_fail(monkeypatch):
    """全源失效必须显式抛错：静默返回空表会把"接口坏了"误报成"休市"。"""
    def fake_get(url):
        raise RealtimeError("HTTP 400")

    monkeypatch.setattr("ashare_quant.realtime._get", fake_get)
    with pytest.raises(RealtimeError) as ei:
        snapshot(["600000"])
    msg = str(ei.value)
    assert "tencent" in msg and "sina" in msg  # 两个源的原因都要有


def test_snapshot_raises_on_empty_payload(monkeypatch):
    """200 但内容为空（例如字段数不足）同样算失败，不能当成"没有这只股票"。"""
    monkeypatch.setattr("ashare_quant.realtime._get", lambda url: "")
    with pytest.raises(RealtimeError):
        snapshot(["600000"])


def test_snapshot_skips_rows_without_prev_close(monkeypatch):
    """昨收缺失/为 0 的行（停牌或脏数据）跳过，避免除零。"""
    payload = _tencent_payload({
        "sh600000": ("浦发银行", 10.0, 0.0),
        "sz000001": ("平安银行", 12.0, 11.9),
    })
    monkeypatch.setattr("ashare_quant.realtime._get", lambda url: payload)
    df = snapshot(["600000", "000001"])
    assert list(df["代码"]) == ["000001"]


def test_snapshot_empty_input_no_network(monkeypatch):
    def boom(url):  # pragma: no cover - 不应被调用
        raise AssertionError("空输入不应发起请求")

    monkeypatch.setattr("ashare_quant.realtime._get", boom)
    df = snapshot([])
    assert df.empty


def test_index_snapshot_returns_four_prefixed(monkeypatch):
    payload = _tencent_payload({
        "sh000001": ("上证指数", 3913.05, 3875.60),
        "sz399001": ("深证成指", 13627.81, 13409.91),
        "sz399006": ("创业板指", 3373.05, 3298.31),
        "sh000300": ("沪深300", 4508.52, 4460.16),
    })
    monkeypatch.setattr("ashare_quant.realtime._get", lambda url: payload)
    df = index_snapshot()
    assert list(df.columns) == ["代码", "名称", "现价", "涨跌幅"]
    assert set(df["代码"]) == set(INDEX_CODES)
    row = df[df["代码"] == "sh000001"].iloc[0]
    assert abs(row["涨跌幅"] - (3913.05 - 3875.60) / 3875.60) < 1e-12


def test_index_snapshot_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr("ashare_quant.realtime._get", lambda url: "")
    with pytest.raises(RealtimeError):
        index_snapshot()
