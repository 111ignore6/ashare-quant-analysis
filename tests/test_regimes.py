import numpy as np
import pandas as pd
from ashare_quant.research.regimes import state_forward_returns, volatility_regimes


def _two_regime_index():
    rng = np.random.default_rng(13)
    idx = pd.date_range("2023-01-02", periods=400, freq="B")
    ret = np.concatenate([rng.normal(0, 0.003, 200), rng.normal(0, 0.02, 200)])
    return pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)


def test_regimes_split_by_volatility():
    idx_close = _two_regime_index()
    reg = volatility_regimes(idx_close, n=10, k=2)
    assert set(reg["state"].dropna().unique()) <= {1, 2}
    assert reg["state"].iloc[:100].mode().iloc[0] == 1
    assert reg["state"].iloc[-100:].mode().iloc[0] == 2


def test_state_forward_returns_has_rows():
    idx_close = _two_regime_index()
    out = state_forward_returns(idx_close, h=10, k=2)
    assert len(out) == 2
    assert {"state", "mean_fwd", "count"} <= set(out.columns)
