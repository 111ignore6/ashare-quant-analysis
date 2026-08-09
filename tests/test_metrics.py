import numpy as np
import pandas as pd
from ashare_quant.backtest.metrics import drawdown_series, metrics_from_returns


def _returns():
    rng = np.random.default_rng(4)
    return pd.Series(rng.normal(0.005, 0.03, 36), index=pd.date_range("2023-01-31", periods=36, freq="ME"))


def test_metrics_keys_and_drawdown():
    r = _returns()
    m = metrics_from_returns(r, periods_per_year=12)
    assert {"annual_return", "annual_vol", "sharpe", "max_drawdown", "win_rate"} <= set(m)
    dd = drawdown_series(r)
    assert (dd <= 0).all()
    assert abs(m["max_drawdown"] - dd.min()) < 1e-9


def test_metrics_of_constant_gain():
    r = pd.Series([0.01] * 12)
    m = metrics_from_returns(r, periods_per_year=12)
    assert m["sharpe"] > 0
    assert m["max_drawdown"] == 0
