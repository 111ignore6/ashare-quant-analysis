import pandas as pd
from ashare_quant.calendar import (TradingCalendar, drop_intraday_today,
                                   market_session)


def test_window_and_contains():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    cal = TradingCalendar.from_dates(dates)
    assert cal.count() == 4
    assert cal.contains("2024-01-03")
    assert not cal.contains("2024-01-06")
    assert len(cal.window("2024-01-03", "2024-01-05")) == 3


def test_dedup_and_sort():
    dates = pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-04"])
    cal = TradingCalendar.from_dates(dates)
    assert cal.all_dates.tolist() == pd.to_datetime(["2024-01-02", "2024-01-04"]).tolist()


def test_market_session_times():
    # 2026-08-11 是周二
    cases = {
        "2026-08-11 08:00": "pre",
        "2026-08-11 10:00": "am",
        "2026-08-11 12:00": "lunch",
        "2026-08-11 14:30": "pm",
        "2026-08-11 16:05": "post",
        "2026-08-08 10:00": "weekend",
    }
    for ts, expected in cases.items():
        assert market_session(ts) == expected, (ts, market_session(ts))


def test_drop_intraday_today(monkeypatch):
    """盘中不允许把"今天"的未收盘行推进；收盘后放行。

    回归：08-12 中午更新把盘中价当收盘写入日线 → 决策日期跳到 08-12，
    实时估值"缺少决策日基准"（面板还是 08-11）。盘中必须挡住当日行。
    """
    # 用实时钟构造"今天/昨天"，避免硬编码日期导致测试随机器日期失效
    today = pd.Timestamp.today().normalize()
    yesterday = today - pd.Timedelta(days=1)
    idx = pd.to_datetime([yesterday.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")])
    df = pd.DataFrame({"close": [9.21, 9.17]}, index=idx)
    monkeypatch.setattr("ashare_quant.calendar.market_session", lambda: "lunch")
    out = drop_intraday_today(df)
    assert list(out.index) == [pd.Timestamp(yesterday)]
    # 收盘后保留当日行
    monkeypatch.setattr("ashare_quant.calendar.market_session", lambda: "post")
    out2 = drop_intraday_today(df)
    assert len(out2) == 2
