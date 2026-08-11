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


def test_modern_models_registered_and_rankable():
    """新模型（XGBoost / LGBMRanker 排序包装）可构造、可拟合、可预测。"""
    from ashare_quant.ml.models import MODELS, list_models

    new_models = ("xgb", "rank_lgb", "rank_xgb", "mlp_deep", "pls", "enet",
                  "huber_lgb", "temporal_decay_lgb", "risk_aware_lgb")
    for name in new_models:
        assert name in list_models()
        factory = MODELS[name]
        model = factory()
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04",
                             "2023-01-05", "2023-01-06", "2023-01-09"]),
             ["A", "B", "C", "D", "E", "F", "G", "H"]], names=["date", "symbol"])
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(len(idx), 4)),
                         index=idx, columns=list("abcd"))
        y = pd.Series(np.random.default_rng(1).normal(size=len(idx)), index=idx)
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(X)
        assert np.isfinite(pred).all()
