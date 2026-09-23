import numpy as np
import pandas as pd

from ashare_quant.ml.evaluate import walk_forward_ml_evaluate
from ashare_quant.ml.features import build_dataset
from ashare_quant.ml.models import linear
from ashare_quant.screening import walk_forward_folds


def _market():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2022-01-03", periods=420, freq="B")
    drift = np.linspace(0.0002, 0.0012, 25)
    rets = rng.normal(0, 0.01, (420, 25)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(25)])
    volume = pd.DataFrame(1000, index=idx, columns=close.columns)
    index_close = close.mean(axis=1)
    return close, volume, index_close


def test_walk_forward_ml_evaluate():
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)
    dates = X.index.get_level_values("date").unique()
    folds = walk_forward_folds(dates, train_months=6, valid_months=3, step_months=3)
    m, series = walk_forward_ml_evaluate("linear", linear, X, y, close,
                                         folds=folds, top_n=5, sample_size=2000)
    assert {"model", "sharpe", "annual_return", "max_drawdown", "mean_ic"} <= set(m)
    assert len(series) > 2
