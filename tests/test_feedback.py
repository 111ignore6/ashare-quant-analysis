import numpy as np
import pandas as pd

from ashare_quant.feedback.log import AdjustmentLog
from ashare_quant.feedback.rotation import rotate_weights


def _returns(n_models=3, n_periods=24, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-31", periods=n_periods, freq="ME")
    cols = {f"m{i}": rng.normal(0.008, 0.03, n_periods) for i in range(n_models)}
    return pd.DataFrame(cols, index=idx)


def test_rotation_weights_normalize_and_threshold():
    r = _returns()
    w = rotate_weights(r, window=6, min_weight=0.05, change_threshold=0.15)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
    assert (w >= 0).all().all()


def test_rotation_no_churn_when_flat():
    r = pd.DataFrame({"m0": [0.01] * 24, "m1": [0.01] * 24})
    w = rotate_weights(r, window=6, change_threshold=0.5)
    assert (w.diff().abs().sum(axis=1) < 0.5).all()


def test_adjustment_log_roundtrip(tmp_path):
    log = AdjustmentLog(tmp_path / "adjust.jsonl")
    log.append(date="2026-08-07", trigger="rotation", action="weights", before={"m0": 0.5}, after={"m1": 0.5}, effect="expected")
    entries = log.read()
    assert len(entries) == 1
    assert entries[0]["trigger"] == "rotation"
