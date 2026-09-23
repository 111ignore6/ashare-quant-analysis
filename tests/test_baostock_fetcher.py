import pandas as pd
from ashare_quant.fetchers.baostock_fetcher import rows_to_frame, to_baostock_code


def test_code_mapping():
    assert to_baostock_code("600000") == "sh.600000"
    assert to_baostock_code("000001") == "sz.000001"
    assert to_baostock_code("688001") == "sh.688001"
    assert to_baostock_code("430047") == "bj.430047"


def test_rows_to_frame():
    fields = ["date", "open", "high", "low", "close", "volume", "amount"]
    rows = [["2024-01-02", "10.0", "11.0", "9.0", "10.5", "1000", "1000000"]]
    df = rows_to_frame(fields, rows)
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert float(df.loc[df.index[0], "close"]) == 10.5
