import numpy as np
import pandas as pd
from ashare_quant.models.candidates import MomentumModel, ReversalModel
from ashare_quant.screening import evaluate, grid_search, run_screening, split_dates


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
    assert "benchmark" in out["model"].tolist()


def test_evaluate_metrics():
    close, volume = _panel()
    tr, va = split_dates(close.index)
    m = evaluate(MomentumModel(20), close, volume, va, top_n=10)
    assert "sharpe" in m and "max_drawdown" in m
