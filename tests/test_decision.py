import json

import numpy as np
import pandas as pd
import pytest
from ashare_quant.ml.decision import decide, load_models, train_and_save
from ashare_quant.ml.features import build_dataset


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


def test_decide_ignores_target_column(tmp_path):
    """特征缓存可能含 target 列（features.parquet 16 列），predict 前必须排除。"""
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)
    train_and_save(X, y, tmp_path, model_names=("lgbm",), sample_size=2000)
    loaded = load_models(tmp_path)
    last_date = X.index.get_level_values("date").max()
    X_with_target = X.copy()
    X_with_target["target"] = 0.0
    picks = decide(loaded, X_with_target, close, last_date, top_n=5)
    assert len(picks) == 5


def test_load_models_broken_library_gives_actionable_error(tmp_path):
    """模型库安装损坏（如 xgboost.dll 缺失）时给出可操作的错误信息，而不是裸 traceback。

    回归：08-27 每日任务在 load_models 崩溃，日志只有深层 pickle traceback。
    """
    (tmp_path / "meta.json").write_text(
        json.dumps({"models": ["xgb"], "thresholds": {}, "trained_on": "2026-08-27"}),
        encoding="utf-8")
    (tmp_path / "xgb.joblib").write_bytes(b"\x00")  # 无法反序列化的损坏文件
    with pytest.raises(RuntimeError, match="xgb") as ei:
        load_models(tmp_path)
    assert "xgboost" in str(ei.value)  # 错误信息给出修复提示（涉及库名）


# —— 训练/校准之间的 embargo 与"死计算"的 thresholds（2026-09-18） ——
class _RecordingModel:
    """只记录"被喂了哪些日期"的桩模型，用来把训练集的边界钉死。"""

    def __init__(self, sink):
        self.sink = sink

    def fit(self, X, y):
        self.sink["fit_dates"] = pd.Index(X.index.get_level_values("date")).unique()
        self.sink["n_fit_rows"] = len(X)
        return self

    def predict(self, X):
        self.sink["n_predict_calls"] = self.sink.get("n_predict_calls", 0) + 1
        self.sink["n_predict_rows"] = self.sink.get("n_predict_rows", 0) + len(X)
        return np.zeros(len(X))


def _stub_models(monkeypatch, sink):
    from ashare_quant.ml import decision as dec
    monkeypatch.setattr(dec, "MODELS", {"stub": lambda: _RecordingModel(sink)})


def test_train_leaves_embargo_between_train_and_calibration(tmp_path, monkeypatch):
    """target 跨度 = horizon 日 ⇒ 训练标签不得伸进校准期（算法问题⑤）。

    旧行为：训练集取到 `calib_cut` 前一天，最后 horizon 个训练日的标签窗口跨进校准期，
    校准期因此不是真正的样本外。现在中间空出 horizon 个交易日（两边都不用）。
    """
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)
    sink: dict = {}
    _stub_models(monkeypatch, sink)
    meta = train_and_save(X, y, tmp_path, model_names=("stub",), horizon=20,
                          sample_size=10**9)

    dates = pd.Index(X.index.get_level_values("date").unique()).sort_values()
    calib_cut = dates[-63]                      # calib_months=3 → 21×3 个交易日
    train_dates = pd.Index(sink["fit_dates"])
    assert train_dates.max() < calib_cut, "训练集不得碰到校准期"
    # 缺口必须正好是 horizon 个交易日（不是 1 天，也不是随便留一点）
    pos_cut, pos_train_end = dates.get_loc(calib_cut), dates.get_loc(train_dates.max())
    assert pos_cut - pos_train_end - 1 == 20, (pos_cut, pos_train_end)
    # 且"最后一个训练日的标签窗口"确实停在 calib_cut 之前（这才是 embargo 的目的）
    assert pos_train_end + 20 < pos_cut
    assert meta["embargo_days"] == 20 and meta["horizon"] == 20


def test_thresholds_are_opt_in(tmp_path, monkeypatch):
    """thresholds 全代码无人消费 ⇒ 默认不计算（算法问题⑥：每轮白跑 6×3 万行预测）。"""
    close, volume, index_close = _market()
    X, y = build_dataset(close, volume, index_close, horizon=20)

    sink: dict = {}
    _stub_models(monkeypatch, sink)
    meta = train_and_save(X, y, tmp_path, model_names=("stub",), horizon=20)
    assert "thresholds" in meta and meta["thresholds"] == {}      # schema 不变
    assert sink.get("n_predict_calls") is None, "默认路径不该在校准期上做预测"

    sink2: dict = {}
    _stub_models(monkeypatch, sink2)
    meta2 = train_and_save(X, y, tmp_path, model_names=("stub",), horizon=20,
                           compute_thresholds=True)
    assert sink2["n_predict_calls"] == 1 and sink2["n_predict_rows"] > 0
    assert "stub" in meta2["thresholds"]
