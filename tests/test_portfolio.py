import json

import pandas as pd

from ashare_quant.portfolio import (append_decision, equity_curve, load_history,
                                    update_portfolio)


def _close():
    idx = pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05",
                          "2026-08-06", "2026-08-07"])
    return pd.DataFrame({
        "600000": [10.0, 10.5, 11.0, 11.5, 12.0],
        "000001": [5.0, 5.2, 5.0, 4.8, 5.0],
    }, index=idx)


def _history():
    return [
        {"date": "2026-08-03", "capital": 100000.0,
         "picks": [{"symbol": "600000", "weight": 0.5},
                   {"symbol": "000001", "weight": 0.5}]},
        {"date": "2026-08-06", "capital": 100000.0,
         "picks": [{"symbol": "600000", "weight": 1.0}]},
    ]


def test_append_and_load_dedup(tmp_path):
    p = tmp_path / "history.jsonl"
    append_decision(p, {"date": "2026-08-03", "picks": []})
    append_decision(p, {"date": "2026-08-05", "picks": []})
    append_decision(p, {"date": "2026-08-03", "picks": [{"symbol": "x"}]})
    h = load_history(p)
    assert [e["date"] for e in h] == ["2026-08-03", "2026-08-05"]
    assert h[0]["picks"] == [{"symbol": "x"}]  # 同日覆盖


def test_equity_curve():
    close = _close()
    rets = equity_curve(close, _history(), capital=100000.0)
    # 段1（08-03 建仓，持到 08-06）：08-04 600000 +5%、000001 +4% → 组合 +4.5%
    # 段2（08-06 起全仓 600000）：08-07 +4.3478%
    assert abs(rets.loc["2026-08-04"] - 0.045) < 1e-9
    assert abs(rets.loc["2026-08-05"] - (11.0 / 10.5 - 1) * 0.5 - (5.0 / 5.2 - 1) * 0.5) < 1e-9
    assert abs(rets.loc["2026-08-07"] - (12.0 / 11.5 - 1)) < 1e-9


def test_update_portfolio_end_to_end(tmp_path):
    class FakeModel:
        def predict(self, rows):
            import numpy as np
            return np.full(len(rows), 0.1)

    class FakeCfg:
        top_n = 2
        initial_capital = 100000.0

    close = _close()
    idx = pd.MultiIndex.from_product([close.index, close.columns],
                                     names=["date", "symbol"])
    X = pd.DataFrame(1.0, index=idx, columns=["f1", "f2"])
    models = {"models": {"m1": FakeModel(), "m2": FakeModel()},
              "meta": {"thresholds": {"m1": 0.05, "m2": 0.05}}}
    summary = update_portfolio(close, FakeCfg(), tmp_path,
                               decision={"date": "2026-08-07", "capital": 100000.0,
                                         "picks": [{"symbol": "600000", "weight": 1.0}]},
                               X=X, models=models)
    assert (tmp_path / "account_equity.csv").exists()
    assert summary["decisions"] >= 1
    assert summary["total_asset"] > 0
    assert (tmp_path / "account_summary.json").exists()
