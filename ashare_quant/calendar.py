from __future__ import annotations

import pandas as pd


class TradingCalendar:
    """由日期序列构造的交易日历（去重、升序）。"""

    def __init__(self, dates) -> None:
        self._dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(dates))))

    @classmethod
    def from_dates(cls, dates) -> "TradingCalendar":
        return cls(dates)

    @property
    def all_dates(self) -> pd.DatetimeIndex:
        return self._dates

    def contains(self, date) -> bool:
        return pd.Timestamp(date) in self._dates

    def window(self, start, end) -> pd.DatetimeIndex:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return self._dates[(self._dates >= s) & (self._dates <= e)]

    def count(self) -> int:
        return len(self._dates)
