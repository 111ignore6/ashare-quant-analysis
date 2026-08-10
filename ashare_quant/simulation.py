from __future__ import annotations

import pandas as pd

from .backtest.engine import run_backtest
from .backtest.metrics import metrics_from_returns
from .backtest.simple import monthly_rebalance_dates
from .feedback.log import AdjustmentLog
from .feedback.rotation import rotate_weights


def run_simulation(models: dict[str, object], close: pd.DataFrame, open_: pd.DataFrame,
                   volume: pd.DataFrame, top_n: int = 50,
                   log_path=None, window: int = 6, threshold: float = 0.15,
                   stop_loss: float | None = None,
                   take_profit: float | None = None) -> dict:
    dates = monthly_rebalance_dates(close.index)
    model_returns = {}
    holdings = {}
    for name, model in models.items():
        score = model.score(close, volume)
        res = run_backtest(score, close, open_, dates, top_n=top_n,
                           stop_loss=stop_loss, take_profit=take_profit)
        model_returns[name] = res.returns
        holdings[name] = res.holdings
    rets = pd.DataFrame(model_returns)
    weights = rotate_weights(rets, window=window, change_threshold=threshold)
    rot = (rets * weights.shift(1)).sum(axis=1, min_count=1).dropna()
    summary_rows = {}
    for name in rets.columns:
        summary_rows[name] = metrics_from_returns(rets[name], periods_per_year=12)
    summary_rows["rotation"] = metrics_from_returns(rot, periods_per_year=12)
    summary = pd.DataFrame(summary_rows).T.reset_index().rename(columns={"index": "model"})
    log = AdjustmentLog(log_path) if log_path else None
    if log is not None:
        first = weights.index[0]
        log.append(date=str(first.date()), trigger="rotation", action="weights",
                   before=dict(weights.iloc[0]), after=dict(weights.iloc[0]), effect="初始化")
        for i in range(1, len(weights)):
            before = dict(weights.iloc[i - 1])
            after = dict(weights.iloc[i])
            if before != after:
                log.append(date=str(weights.index[i].date()), trigger="rotation", action="weights",
                           before=before, after=after, effect="滚动夏普轮动")
    return {"model_returns": rets, "rotation_returns": rot, "weights": weights,
            "summary": summary, "holdings": holdings, "log": log}
