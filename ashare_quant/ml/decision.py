"""每日决策引擎：训练并持久化模型，按最新数据生成模拟投资决策。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .models import MODELS


def train_and_save(X: pd.DataFrame, y: pd.Series, out_dir: Path,
                   model_names=("lgbm", "histgb", "rf", "svm", "knn", "linear"),
                   sample_size: int = 60000, calib_months: int = 3,
                   alpha: float = 0.5, svm_sample_cap: int = 20000,
                   calib_sample_cap: int = 30000, as_of=None,
                   horizon: int = 20, compute_thresholds: bool = False) -> dict:
    """在历史上训练若干模型；最后 calib_months 作为校准期（阈值可选）。

    **训练集与校准期之间留 embargo（2026-09-18 修，见 AGENTS.md 算法问题⑤）**：
    target 是"未来 horizon 日收益"，相邻交易日的标签窗口彼此重叠。旧实现把训练集一直
    取到 `calib_cut` 前一天，于是训练标签会**伸进校准期**（最多 horizon 个交易日）——
    校准期的"样本外残差"因此偏乐观，而它正是 `thresholds` 的度量口径。现在训练集右端
    再往前退 horizon 个交易日（这一段的标签两边都不用，即 purged/embargo 的标准做法），
    `meta["embargo_days"]` 记录实际留出的交易日数。

    **thresholds 默认不再计算（2026-09-18，见算法问题⑥）**：`thresholds` 从写进
    `decision.json` / 账户历史起就没有任何消费方（全代码 0 处读取），而它每轮要为每个
    模型各预测最多 `calib_sample_cap` 行。字段保留（schema 不变，默认 `{}`），
    真要用时传 `compute_thresholds=True`。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.Index(X.index.get_level_values("date").unique()).sort_values()
    n_calib = int(21 * calib_months)
    calib_cut = dates[-n_calib] if len(dates) > n_calib else dates[0]
    # 标签窗口 = (d, d+horizon]，所以"标签不越过校准期"要求 d 比 calib_cut 至少早
    # horizon+1 个交易日；等价于训练集右端退到 dates[pos - horizon] 之前。
    pos = int(dates.get_loc(calib_cut))
    train_cut_pos = max(0, pos - max(0, int(horizon)))
    train_cut = dates[train_cut_pos]
    embargo_days = int(pos - train_cut_pos)
    day = X.index.get_level_values("date")
    train_mask = day < train_cut
    calib_mask = day >= calib_cut
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
        if compute_thresholds:
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
        "horizon": int(horizon),
        "embargo_days": embargo_days,
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
    models: dict[str, object] = {}
    for name in meta["models"]:
        path = out_dir / f"{name}.joblib"
        try:
            models[name] = joblib.load(path)
        except Exception as exc:
            # 历史上遇到：xgboost.dll 缺失导致 xgb.joblib 加载失败，整个
            # 每日决策阶段崩溃且日志只有深层 pickle traceback，无法定位。
            # 保持 fail-fast（决策不允许悄悄缺模型），但给出明确修复路径。
            raise RuntimeError(
                f"模型 {name} 加载失败: {path}\n"
                f"原因: {exc}\n"
                "通常是该模型对应的机器学习库安装损坏或版本不兼容（例如 "
                "xgboost 缺少 xgboost.dll）。修复: python -m pip install "
                "--force-reinstall --no-deps xgboost 或 lightgbm / "
                "scikit-learn（以报错提到的库为准）。"
            ) from exc
    return {"models": models, "meta": meta}


def decide(models: dict, X: pd.DataFrame, close: pd.DataFrame,
           date, top_n: int = 50) -> pd.DataFrame:
    """在指定日期用已训练模型打分 → 截面排名均值融合 → 选 Top-N（等权）。

    排名融合：每个模型预测转当日横截面百分位排名（0-1）再平均。不同模型的
    量纲/极端值不再干扰融合，直接对齐 Top-N 选股目标（rank_ensemble 在全
    市场 benchmark 中年化 77.9%）。`score` 列保留原始预测均值（未来 20 日
    收益预期，供展示），排序以 `rank_score` 为准。
    """
    rows = X[X.index.get_level_values("date") == pd.Timestamp(date)]
    if rows.empty:
        raise ValueError(f"日期 {date} 无特征数据")
    # 特征缓存可能含 target 列，预测前排除（模型按特征列训练）
    rows = rows.drop(columns=[c for c in ("target",) if c in rows.columns])
    preds = pd.DataFrame({name: m.predict(rows) for name, m in models["models"].items()},
                         index=rows.index)
    model_names = list(preds.columns)
    ranks = preds.groupby(level="date").rank(pct=True)
    rank_score = ranks.mean(axis=1)
    # 展示用"预期收益"：取各模型预测的中位数（对量纲差异稳健，
    # 排序/选股仍以 rank_score 为准）
    display_pred = np.median(preds.to_numpy(), axis=1)
    table = pd.DataFrame({
        "symbol": rank_score.index.get_level_values("symbol"),
        "rank_score": rank_score.values,
        "score": display_pred,
    }).sort_values(["rank_score", "score"], ascending=False)
    picks = table.head(top_n).copy()
    # 决策解释：记录每只股票的各模型预测明细
    picks["model_scores"] = [
        {name: round(float(preds.loc[(pd.Timestamp(date), sym), name]), 4)
         for name in model_names}
        for sym in picks["symbol"]
    ]
    picks["weight"] = 1.0 / len(picks)
    return picks.reset_index(drop=True)
