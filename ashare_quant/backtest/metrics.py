from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns.fillna(0)).cumprod()
    peak = equity.cummax()
    return equity / peak - 1


def metrics_from_returns(returns: pd.Series, periods_per_year: float = 12) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0,
                "profit_loss_ratio": float("nan"), "calmar": float("nan")}
    ann_return = (1 + r).prod() ** (periods_per_year / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    max_dd = float(drawdown_series(r).min())
    gains = r[r > 0]
    losses = r[r < 0]
    plr = float(gains.mean() / abs(losses.mean())) if len(losses) else float("nan")
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else float("nan")
    return {
        "annual_return": float(ann_return),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "win_rate": float((r > 0).mean()),
        "profit_loss_ratio": plr,
        "calmar": calmar,
    }
