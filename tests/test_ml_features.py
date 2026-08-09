import numpy as np
import pandas as pd
from ashare_quant.ml.features import build_dataset


def _market():
    rng = np.random.default_rng(12)
    idx = pd.date_range("2023-01-02", periods=160, freq="B")
    drift = np.linspace(0.0002, 0.001, 10)
    rets = rng.normal(0, 0.01, (160, 10)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(10)])
    volume = pd.DataFrame(1000, index=idx, columns=close.columns)
    index_close = close.mean(axis=1)
    return close, volume, index_close


def test_build_dataset_shape_and_target():
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)
    assert X.index.names == ["date", "symbol"]
    assert "ret_20" in X.columns and "index_state" in X.columns
    sample = X.index[100]
    d, s = sample
    assert abs(y.loc[sample] - (close.loc[close.index[close.index.get_loc(d) + 20], s] / close.loc[d, s] - 1)) < 1e-9
