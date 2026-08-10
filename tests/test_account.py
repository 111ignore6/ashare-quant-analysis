import pandas as pd

from ashare_quant.account import account_snapshot


def _decision():
    return {
        "date": "2026-08-07",
        "initial_capital": 100000.0,
        "picks": [
            {"symbol": "600000", "weight": 0.5, "score": 0.1},
            {"symbol": "000001", "weight": 0.5, "score": 0.2},
        ],
    }


def _panel():
    idx = pd.to_datetime(["2026-08-07", "2026-08-10"])
    return pd.DataFrame(
        {"600000": [10.0, 11.0], "000001": [5.0, 4.5]}, index=idx)


def test_account_snapshot():
    panel = _panel()
    acc = account_snapshot(_decision(), panel)
    assert acc is not None
    # 初始资金 10 万，各 50% → 每只 5 万
    rows = acc["rows"].set_index("代码")
    assert rows.loc["600000", "投入金额"] == 50000.0
    assert rows.loc["600000", "股数"] == 5000.0  # 5万 / 10元
    # 600000 涨 10%，000001 跌 10%
    assert acc["total_asset"] == 50000 * 1.1 + 50000 * 0.9
    assert abs(acc["total_return"]) < 1e-9
    assert acc["total_pnl"] == 0.0
    assert acc["as_of"] == "2026-08-10"


def test_account_snapshot_with_prices():
    panel = _panel()
    prices = pd.Series({"600000": 12.0, "000001": 5.0})
    acc = account_snapshot(_decision(), panel, prices=prices)
    assert acc["total_asset"] == 50000 * 12 / 10 + 50000 * 5 / 5
    assert acc["total_return"] == acc["total_asset"] / 100000 - 1


def test_account_snapshot_none_without_picks():
    assert account_snapshot({"picks": []}, _panel()) is None
    assert account_snapshot({"picks": [{"symbol": "x", "weight": 1}]},
                            pd.DataFrame()) is None
