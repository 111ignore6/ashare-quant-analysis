import numpy as np
import pandas as pd
from ashare_quant.research.stats import distribution_stats, volatility_clustering


def _close_panel(n_days=1000, n_stocks=50, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    cols = {f"S{i:04d}": 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days))) for i in range(n_stocks)}
    return pd.DataFrame(cols, index=idx)


def test_distribution_stats_reports_kurtosis():
    close = _close_panel()
    dist = distribution_stats(close)
    assert "kurtosis" in dist
    assert dist["skewness"] is not None
    assert dist["n_symbols"] == 50


def test_volatility_clustering_on_garch_like_series():
    rng = np.random.default_rng(7)
    n = 2000
    ret = np.zeros(n)
    sigma2 = np.ones(n) * 1e-4
    for t in range(1, n):
        sigma2[t] = 1e-5 + 0.9 * ret[t - 1] ** 2 + 0.08 * sigma2[t - 1]
        ret[t] = rng.normal(0, np.sqrt(sigma2[t]))
    out = volatility_clustering(pd.Series(ret))
    assert out["ljungbox_p"] < 0.05
    assert out["arch_p"] < 0.05
