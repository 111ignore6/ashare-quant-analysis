import numpy as np
import pandas as pd
from ashare_quant.research.factor_stats import (cross_sectional_ic, factor_report,
                                                forward_returns, layer_returns, stability_by_year)


def _momentum_panel():
    rng = np.random.default_rng(9)
    idx = pd.date_range("2023-01-02", periods=250, freq="B")
    drift = np.linspace(0.0001, 0.002, 40)
    rets = rng.normal(0, 0.01, (250, 40)) + drift
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                        columns=[f"S{i:04d}" for i in range(40)])


def test_cross_sectional_ic_positive_for_momentum():
    close = _momentum_panel()
    factor = close.pct_change(20)
    target = forward_returns(close, 20)
    ic = cross_sectional_ic(factor, target)
    assert ic.mean() > 0.05


def test_layer_returns_monotonic():
    close = _momentum_panel()
    factor = close.pct_change(20)
    target = forward_returns(close, 20)
    layers = layer_returns(factor, target, k=5)
    assert len(layers) == 5
    assert layers.iloc[-1] > layers.iloc[0]


def test_stability_by_year_and_report():
    close = _momentum_panel()
    rep = factor_report(close, volume=pd.DataFrame(1000, index=close.index, columns=close.columns))
    assert "ic_summary" in rep and "ic_series" in rep
    assert "momentum" in rep["ic_summary"].index
    assert list(stability_by_year(rep["ic_series"]["momentum"]).columns) == ["mean_ic", "hit_rate", "count"]
