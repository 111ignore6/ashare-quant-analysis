import pandas as pd
from ashare_quant.calendar import TradingCalendar
from ashare_quant.validation import validate_symbol


def _df(dates, close):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": close, "high": [c + 0.1 for c in close], "low": [c - 0.1 for c in close],
         "close": close, "volume": [1000] * len(idx), "amount": [1e6] * len(idx)},
        index=idx,
    )


def test_duplicate_and_nonpositive():
    df = _df(["2024-01-02", "2024-01-02"], [10, 0])
    issues = validate_symbol("000001", df, None)
    rules = {i.rule for i in issues}
    assert "duplicate_dates" in rules
    assert "nonpositive_price" in rules


def test_missing_dates_exceeds_threshold():
    cal = TradingCalendar.from_dates(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
    df = _df(["2024-01-02", "2024-01-05"], [10, 11])
    issues = validate_symbol("000001", df, cal, missing_threshold=0.1)
    assert any(i.rule == "missing_dates" for i in issues)


def test_empty_flagged():
    issues = validate_symbol("000001", None, None)
    assert issues[0].rule == "empty"
