from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.metrics import metrics_from_returns
from ..backtest.simple import monthly_rebalance_dates, simple_topn_returns
from ..research.factor_stats import cross_sectional_ic
from ..screening import walk_forward_folds


def predict_scores(model, X: pd.DataFrame, dates) -> pd.DataFrame:
    rows = X[X.index.get_level_values("date").isin(dates)]
    preds = pd.Series(model.predict(rows), index=rows.index)
    return preds.unstack("symbol")


def _rank_ic(preds: pd.Series, y: pd.Series) -> pd.Series:
    """预测分与真实收益的逐日 Spearman IC（向量化，等价于逐日 corr）。"""
    pred_w = preds.unstack("symbol")
    actual_w = y[preds.index].unstack("symbol")
    return cross_sectional_ic(pred_w, actual_w, min_n=2)


def walk_forward_ml_evaluate(name: str, make_model, X: pd.DataFrame, y: pd.Series,
                             close: pd.DataFrame, folds=None, top_n: int = 50,
                             sample_size: int = 40000) -> tuple[dict, pd.Series]:
    if folds is None:
        dates = X.index.get_level_values("date").unique()
        folds = walk_forward_folds(dates)
    all_rets, all_ics = [], []
    for tr_dates, va_dates in folds:
        mask = X.index.get_level_values("date").isin(tr_dates)
        idx = np.flatnonzero(mask)
        if len(idx) > sample_size:
            idx = np.random.default_rng(0).choice(idx, sample_size, replace=False)
        model = make_model()
        model.fit(X.iloc[idx], y.iloc[idx])
        rdates = monthly_rebalance_dates(va_dates)
        score = predict_scores(model, X, rdates)
        rets = simple_topn_returns(score, close, rdates, top_n=top_n)
        all_rets.append(rets)
        va_rows = X[X.index.get_level_values("date").isin(rdates)]
        preds = pd.Series(model.predict(va_rows), index=va_rows.index)
        ic = _rank_ic(preds, y[va_rows.index])
        all_ics.append(ic)
    series = pd.concat(all_rets)
    metrics = metrics_from_returns(series, periods_per_year=12)
    metrics["model"] = name
    metrics["mean_ic"] = float(pd.concat(all_ics).mean()) if all_ics else 0.0
    return metrics, series
