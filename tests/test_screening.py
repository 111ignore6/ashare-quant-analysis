import numpy as np
import pandas as pd

from ashare_quant.models.candidates import MomentumModel, ReversalModel
from ashare_quant.screening import (
    evaluate,
    grid_search,
    run_screening,
    split_dates,
    walk_forward_evaluate,
    walk_forward_folds,
)


def _panel():
    rng = np.random.default_rng(8)
    idx = pd.date_range("2022-01-03", periods=500, freq="B")
    drift = np.linspace(0.0002, 0.0015, 30)
    rets = rng.normal(0, 0.01, (500, 30)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(30)])
    volume = pd.DataFrame(1000, index=idx, columns=close.columns)
    return close, volume


def test_split_dates():
    close, _ = _panel()
    tr, va = split_dates(close.index, train_frac=0.6)
    assert len(tr) > len(va)
    assert tr[-1] < va[0]


def test_walk_forward_folds_and_evaluate():
    close, volume = _panel()
    folds = walk_forward_folds(close.index, train_months=6, valid_months=3, step_months=3)
    assert len(folds) >= 2
    for tr, va in folds:
        assert tr[-1] < va[0]
    series, m, params = walk_forward_evaluate(ReversalModel, {"horizon": [20, 60]},
                                              close, volume, folds, top_n=10)
    assert len(series) >= 3
    assert "sharpe" in m
    assert "horizon" in params


def test_grid_search_picks_best_train_sharpe():
    close, volume = _panel()
    tr, _ = split_dates(close.index)
    best = grid_search(ReversalModel, {"horizon": [20, 60]}, close, volume, tr)
    assert "horizon" in best and "sharpe" in best


def test_run_screening_returns_decisions():
    close, volume = _panel()
    bench = close.mean(axis=1)
    out = run_screening(close, volume, bench, top_n=10)
    assert {"model", "keep", "reason"} <= set(out.columns)
    assert any("benchmark" in m for m in out["model"].tolist())


def test_evaluate_metrics():
    close, volume = _panel()
    tr, va = split_dates(close.index)
    m = evaluate(MomentumModel(20), close, volume, va, top_n=10)
    assert "sharpe" in m and "max_drawdown" in m


def test_covered_dates_drops_collapsed_days():
    """等权全市场基准必须剔除横截面塌缩的日期（09-16 曾据此算出 -57% 假暴跌）。

    月频基准同样受伤：一个月频观测被污染就足以改变夏普与"是否跑赢基准"的判断。
    """
    from ashare_quant.screening import covered_dates

    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    close = pd.DataFrame({f"{i:06d}": [10.0] * 6 for i in range(40)}, index=idx)
    collapsed_day = idx[4]
    close.loc[collapsed_day, [f"{i:06d}" for i in range(3, 40)]] = float("nan")

    kept = covered_dates(close, list(idx))
    assert collapsed_day not in kept
    assert len(kept) == 5
    # 覆盖正常时一天都不剔除
    assert covered_dates(close.fillna(10.0), list(idx)) == list(idx)
