import numpy as np
import pandas as pd

from ashare_quant.research.factors import compute_factors, winsorize_zscore


def _panel(n_days=120, n_stocks=30, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    cols = {f"S{i:04d}": 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days))) for i in range(n_stocks)}
    return pd.DataFrame(cols, index=idx)


def test_compute_factors_shapes():
    close = _panel()
    volume = pd.DataFrame(1000, index=close.index, columns=close.columns)
    factors = compute_factors(close, volume)
    assert set(factors) == {"momentum", "reversal", "volatility", "ma_deviation", "volume_ratio"}
    assert factors["momentum"].shape == close.shape


def test_winsorize_zscore_cross_sectional():
    df = pd.DataFrame({"a": [1.0, 2.0, 100.0], "b": [2.0, 4.0, 200.0]}, index=[0, 1, 2])
    z = winsorize_zscore(df)
    row0 = z.loc[0]
    assert abs(row0.mean()) < 1e-9
    assert abs(row0.std(ddof=0) - 1) < 1e-9
