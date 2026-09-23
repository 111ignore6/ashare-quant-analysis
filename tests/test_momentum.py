import numpy as np
import pandas as pd

from ashare_quant.research.momentum import horizon_scan


def test_momentum_scan_detects_positive_ic():
    rng = np.random.default_rng(3)
    n_days, n_stocks = 300, 40
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    drift = np.linspace(0.0001, 0.002, n_stocks)  # 强者恒强：高漂移股票持续跑赢
    rets = rng.normal(0, 0.01, (n_days, n_stocks)) + drift
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(n_stocks)])
    out = horizon_scan(close, horizons=(5, 20, 60))
    assert list(out["horizon"]) == [5, 20, 60]
    assert out.loc[out["horizon"] == 60, "mean_ic"].iloc[0] > 0.05
