import numpy as np
import pandas as pd

from ashare_quant.backtest.metrics import drawdown_series, metrics_from_returns


def _returns():
    rng = np.random.default_rng(4)
    return pd.Series(rng.normal(0.005, 0.03, 36), index=pd.date_range("2023-01-31", periods=36, freq="ME"))


def test_metrics_keys_and_drawdown():
    r = _returns()
    m = metrics_from_returns(r, periods_per_year=12)
    assert {"annual_return", "annual_vol", "sharpe", "max_drawdown", "win_rate",
            "profit_loss_ratio", "calmar"} <= set(m)
    dd = drawdown_series(r)
    assert (dd <= 0).all()
    assert abs(m["max_drawdown"] - dd.min()) < 1e-9


def test_metrics_of_constant_gain():
    r = pd.Series([0.01] * 12)
    m = metrics_from_returns(r, periods_per_year=12)
    assert m["sharpe"] > 0
    assert m["max_drawdown"] == 0


def test_profit_loss_ratio():
    r = pd.Series([0.05, -0.02, 0.03, -0.01])
    m = metrics_from_returns(r, periods_per_year=12)
    assert abs(m["profit_loss_ratio"] - (0.04 / 0.015)) < 1e-9


def test_plr_and_calmar_nan_on_no_losses():
    import numpy as np
    m = metrics_from_returns(pd.Series([0.01] * 12), periods_per_year=12)
    assert np.isnan(m["profit_loss_ratio"])
    assert np.isnan(m["calmar"])
