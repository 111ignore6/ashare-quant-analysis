import numpy as np
import pandas as pd
from ashare_quant.ml.decision import decide, load_models, train_and_save
from ashare_quant.ml.features import build_dataset
from ashare_quant.ml.models import linear


def _market():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-02", periods=300, freq="B")
    drift = np.linspace(0.0003, 0.001, 20)
    rets = rng.normal(0, 0.01, (300, 20)) + drift
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                         columns=[f"S{i:04d}" for i in range(20)])
    volume = pd.DataFrame(1000, index=idx, columns=close.columns)
    index_close = close.mean(axis=1)
    return close, volume, index_close


def test_train_decide_roundtrip(tmp_path):
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)
    meta = train_and_save(X, y, tmp_path, model_names=("lgbm",), sample_size=2000)
    assert "thresholds" in meta
    loaded = load_models(tmp_path)
    last_date = X.index.get_level_values("date").max()
    picks = decide(loaded, X, close, last_date, top_n=5)
    assert len(picks) == 5
    assert abs(picks["weight"].sum() - 1.0) < 1e-9
