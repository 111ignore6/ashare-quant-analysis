import numpy as np
import pandas as pd
from ashare_quant.models.candidates import LowVolModel, MomentumModel, MultiFactorModel, ReversalModel


def _panel(n_days=150, n_stocks=30, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    drift = np.linspace(-0.0005, 0.001, n_stocks)
    rets = rng.normal(0, 0.02, (n_days, n_stocks)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(n_stocks)])
    volume = pd.DataFrame(1000 + rng.normal(0, 100, close.shape), index=idx, columns=close.columns)
    return close, volume


def test_models_return_score_panels():
    close, volume = _panel()
    for model in [ReversalModel(60), LowVolModel(20), MomentumModel(20), MultiFactorModel()]:
        score = model.score(close, volume)
        assert isinstance(score, pd.DataFrame)
        assert score.shape == close.shape
        assert score.columns.tolist() == close.columns.tolist()


def test_reversal_scores_oversold_higher():
    close, volume = _panel()
    model = ReversalModel(20)
    score = model.score(close, volume)
    last = score.iloc[-1].dropna()
    worst = close.pct_change(20, fill_method=None).iloc[-1].idxmin()
    assert last.idxmax() == worst
