import pandas as pd
from ashare_quant.calendar import TradingCalendar


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
