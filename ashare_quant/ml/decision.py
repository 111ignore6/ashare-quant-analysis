"""每日决策引擎：训练并持久化模型，按最新数据生成模拟投资决策。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .models import MODELS


def train_and_save(X: pd.DataFrame, y: pd.Series, out_dir: Path,
                   model_names=("lgbm", "histgb", "svm"),
                   sample_size: int = 60000, calib_months: int = 3,
                   alpha: float = 0.5, svm_sample_cap: int = 20000,
                   calib_sample_cap: int = 30000, as_of=None) -> dict:
    """在全部历史上训练若干模型；最后 calib_months 作为校准期计算残差阈值。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = X.index.get_level_values("date").unique()
    calib_cut = dates[-int(21 * calib_months):][0] if len(dates) > 21 * calib_months else dates[0]
    train_mask = X.index.get_level_values("date") < calib_cut
    calib_mask = X.index.get_level_values("date") >= calib_cut
    idx = np.flatnonzero(train_mask)
    if len(idx) > sample_size:
        idx = np.random.default_rng(0).choice(idx, sample_size, replace=False)
    Xtr, ytr = X.iloc[idx], y.iloc[idx]
    thresholds: dict[str, float] = {}
    for name in model_names:
        model = MODELS[name]()
        if name == "svm" and len(idx) > svm_sample_cap:
            svm_idx = np.random.default_rng(0).choice(idx, svm_sample_cap, replace=False)
            model.fit(X.iloc[svm_idx], y.iloc[svm_idx])
        else:
            model.fit(Xtr, ytr)
        joblib.dump(model, out_dir / f"{name}.joblib")
        calib_idx = np.flatnonzero(calib_mask)
        if len(calib_idx) > calib_sample_cap:
            calib_idx = np.random.default_rng(1).choice(
                calib_idx, calib_sample_cap, replace=False)
        if len(calib_idx):
            resid = np.abs(model.predict(X.iloc[calib_idx]) - y.iloc[calib_idx].values)
            thresholds[name] = float(np.quantile(resid, 1 - alpha))
    meta = {
        "models": list(model_names),
        "thresholds": thresholds,
        "trained_on": (str(pd.Timestamp(as_of).date())
                       if as_of is not None else str(dates.max().date())),
        "alpha": alpha,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return meta


def load_models(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    models = {name: joblib.load(out_dir / f"{name}.joblib") for name in meta["models"]}
    return {"models": models, "meta": meta}


def decide(models: dict, X: pd.DataFrame, close: pd.DataFrame,
           date, top_n: int = 50) -> pd.DataFrame:
    """在指定日期用已训练模型打分 → 置信度软加权 → 选 Top-N（等权）。"""
    rows = X[X.index.get_level_values("date") == pd.Timestamp(date)]
    if rows.empty:
        raise ValueError(f"日期 {date} 无特征数据")
    preds = pd.DataFrame({name: m.predict(rows) for name, m in models["models"].items()},
                         index=rows.index)
    model_names = list(preds.columns)
    mean_pred = preds.mean(axis=1)
    threshold = np.mean(list(models["meta"]["thresholds"].values()))
    if threshold > 0:
        conf = (mean_pred.abs() / threshold).clip(upper=1.0)
        score = mean_pred * conf
    else:
        score = mean_pred
    table = pd.DataFrame({"symbol": score.index.get_level_values("symbol"),
                          "score": score.values}).sort_values("score", ascending=False)
    picks = table.head(top_n).copy()
    # 决策解释：记录每只股票的各模型预测明细
    picks["model_scores"] = [
        {name: round(float(preds.loc[(pd.Timestamp(date), sym), name]), 4)
         for name in model_names}
        for sym in picks["symbol"]
    ]
    picks["weight"] = 1.0 / len(picks)
    return picks.reset_index(drop=True)
