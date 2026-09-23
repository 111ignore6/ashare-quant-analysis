import numpy as np
import pandas as pd
from ashare_quant.models.candidates import MomentumModel, ReversalModel
from ashare_quant.simulation import run_simulation


def _market():
    rng = np.random.default_rng(9)
    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    drift = np.linspace(0.0003, 0.0012, 20)
    rets = rng.normal(0, 0.01, (300, 20)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(20)])
    open_ = close.shift(1).fillna(close)
    volume = pd.DataFrame(1000, index=idx, columns=close.columns)
    return close, open_, volume


def test_run_simulation_returns_models_and_rotation(tmp_path):
    close, open_, volume = _market()
    models = {"momentum": MomentumModel(20), "reversal": ReversalModel(20)}
    out = run_simulation(models, close, open_, volume, top_n=5, log_path=tmp_path / "adjust.jsonl")
    assert set(out["model_returns"].columns) == {"momentum", "reversal"}
    assert {"model", "sharpe", "annual_return", "max_drawdown"} <= set(out["summary"].columns)
    assert out["weights"] is not None
    assert len(out["log"].read()) >= 1
