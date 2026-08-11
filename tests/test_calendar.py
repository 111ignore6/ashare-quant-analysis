import pandas as pd
from ashare_quant.calendar import TradingCalendar, market_session


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
