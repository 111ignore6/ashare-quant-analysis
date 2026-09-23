import numpy as np
import pandas as pd

from ashare_quant.research.redundancy import factor_correlation, pca_redundancy


def _duplicated_factors():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-02", periods=100, freq="B")
    base = pd.DataFrame(rng.normal(0, 1, (100, 20)), index=idx,
                        columns=[f"S{i:04d}" for i in range(20)])
    return {"a": base, "b": base.copy(), "c": base + rng.normal(0, 0.5, base.shape)}


def test_factor_correlation_high_for_copies():
    corr = factor_correlation(_duplicated_factors())
    assert corr.loc["a", "b"] > 0.99


def test_pca_first_component_dominant():
    out = pca_redundancy(_duplicated_factors())
    assert out["first_ratio"] > 0.8
    assert out["n_components_80"] == 1
