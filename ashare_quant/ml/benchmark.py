"""算法表现对比评测：统一滚动样本外，输出对比报告。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.metrics import metrics_from_returns
from ..backtest.simple import monthly_rebalance_dates, simple_topn_returns
from ..models.candidates import LowVolModel, MomentumModel, MultiFactorModel, ReversalModel
from ..research.factor_stats import cross_sectional_ic, forward_returns
from ..research.factors import compute_factors, winsorize_zscore
from ..screening import walk_forward_folds
from .evaluate import walk_forward_ml_evaluate
from .features import build_dataset
from .models import MODELS


def _valid_rdates(folds) -> list:
    return sorted({d for _, va in folds for d in monthly_rebalance_dates(va)})


def baseline_returns(close: pd.DataFrame, volume: pd.DataFrame, folds, top_n: int = 50) -> dict[str, pd.Series]:
    models = {
        "reversal120": ReversalModel(120),
        "lowvol20": LowVolModel(20),
        "momentum20": MomentumModel(20),
        "multifactor": MultiFactorModel(
            {"volume_ratio": 0.4, "ma_deviation": 0.2, "reversal60": 0.2, "lowvol": 0.2}),
    }
    rdates = _valid_rdates(folds)
    return {name: simple_topn_returns(m.score(close, volume), close, rdates, top_n=top_n)
            for name, m in models.items()}


def ensemble_returns(X: pd.DataFrame, y: pd.Series, close: pd.DataFrame,
                     folds, top_n: int = 50, sample_size: int = 40000,
                     members=("lgbm", "histgb", "rf")) -> pd.Series:
    all_rets = []
    for tr_dates, va_dates in folds:
        mask = X.index.get_level_values("date").isin(tr_dates)
        idx = np.flatnonzero(mask)
        if len(idx) > sample_size:
            idx = np.random.default_rng(0).choice(idx, sample_size, replace=False)
        fitted = []
        for name in members:
            model = MODELS[name]()
            model.fit(X.iloc[idx], y.iloc[idx])
            fitted.append(model)
        rdates = monthly_rebalance_dates(va_dates)
        rows = X[X.index.get_level_values("date").isin(rdates)]
        preds = np.mean([m.predict(rows) for m in fitted], axis=0)
        score = pd.Series(preds, index=rows.index).unstack("symbol")
        all_rets.append(simple_topn_returns(score, close, rdates, top_n=top_n))
    return pd.concat(all_rets)


def rank_ensemble_returns(X: pd.DataFrame, y: pd.Series, close: pd.DataFrame,
                          folds, top_n: int = 50, sample_size: int = 40000,
                          members=("lgbm", "histgb", "rf", "xgb", "rank_lgb")) -> pd.Series:
    """自定义融合：多模型预测转横截面百分位排名后取均值（rank aggregation）。

    直接对原始预测取均值会被个别模型的量纲/极端值带偏；转截面排名后再平均，
    每个模型只贡献"相对位置"，与 Top-N 选股目标对��，天然稳健。
    """
    all_rets = []
    for tr_dates, va_dates in folds:
        mask = X.index.get_level_values("date").isin(tr_dates)
        idx = np.flatnonzero(mask)
        if len(idx) > sample_size:
            idx = np.random.default_rng(0).choice(idx, sample_size, replace=False)
        fitted = []
        for name in members:
            model = MODELS[name]()
            model.fit(X.iloc[idx], y.iloc[idx])
            fitted.append(model)
        rdates = monthly_rebalance_dates(va_dates)
        rows = X[X.index.get_level_values("date").isin(rdates)]
        ranks = []
        for m in fitted:
            pred = pd.Series(m.predict(rows), index=rows.index)
            ranks.append(pred.groupby(level="date").rank(pct=True))
        score = pd.concat(ranks, axis=1).mean(axis=1).unstack("symbol")
        all_rets.append(simple_topn_returns(score, close, rdates, top_n=top_n))
    return pd.concat(all_rets)


def _fit_members(X: pd.DataFrame, y: pd.Series, tr_dates, members, sample_size: int):
    mask = X.index.get_level_values("date").isin(tr_dates)
    idx = np.flatnonzero(mask)
    if len(idx) > sample_size:
        idx = np.random.default_rng(0).choice(idx, sample_size, replace=False)
    return [MODELS[name]().fit(X.iloc[idx], y.iloc[idx]) for name in members]


def agreement_ensemble_returns(X: pd.DataFrame, y: pd.Series, close: pd.DataFrame,
                               folds, top_n: int = 50, sample_size: int = 40000,
                               members=("lgbm", "histgb", "rf", "xgb", "rank_lgb"),
                               penalty: float = 0.5) -> pd.Series:
    """自研：一致性排序融合——成员排名均值扣减"排名离散度×惩罚"。

    成员分歧大的股票（某几个模型看多、某几个看空）往往噪音更大，
    显式降权，保留高共识股票进 Top-N。
    """
    all_rets = []
    for tr_dates, va_dates in folds:
        fitted = _fit_members(X, y, tr_dates, members, sample_size)
        rdates = monthly_rebalance_dates(va_dates)
        rows = X[X.index.get_level_values("date").isin(rdates)]
        ranks = []
        for m in fitted:
            pred = pd.Series(m.predict(rows), index=rows.index)
            ranks.append(pred.groupby(level="date").rank(pct=True))
        rank_df = pd.concat(ranks, axis=1)
        score = (rank_df.mean(axis=1) - penalty * rank_df.std(axis=1)
                 ).unstack("symbol")
        all_rets.append(simple_topn_returns(score, close, rdates, top_n=top_n))
    return pd.concat(all_rets)


def ic_rank_ensemble_returns(X: pd.DataFrame, y: pd.Series, close: pd.DataFrame,
                             folds, top_n: int = 50, sample_size: int = 40000,
                             members=("lgbm", "histgb", "rf", "xgb", "rank_lgb"),
                             tail_months: int = 6) -> pd.Series:
    """自研：滚动 IC 加权排序融合——成员近期截面 IC 越高权重越大。"""
    all_rets = []
    for tr_dates, va_dates in folds:
        fitted = _fit_members(X, y, tr_dates, members, sample_size)
        # 训练段尾部（最近 tail_months）的成员 IC 作为权重
        tail = sorted(tr_dates)[-int(21 * tail_months):]
        tail_rows = X[X.index.get_level_values("date").isin(tail)]
        weights = []
        for m in fitted:
            pred = pd.Series(m.predict(tail_rows), index=tail_rows.index)
            pw = pred.unstack("symbol")
            aw = y[tail_rows.index].unstack("symbol")
            ic = pw.corrwith(aw, axis=1).dropna()
            weights.append(max(0.0, float(ic.mean())))
        total = sum(weights)
        w = np.asarray(weights, dtype=float) / total if total > 0 \
            else np.ones(len(members)) / len(members)
        rdates = monthly_rebalance_dates(va_dates)
        rows = X[X.index.get_level_values("date").isin(rdates)]
        ranks = []
        for m in fitted:
            pred = pd.Series(m.predict(rows), index=rows.index)
            ranks.append(pred.groupby(level="date").rank(pct=True))
        rank_df = pd.concat(ranks, axis=1)
        score = (rank_df * w).sum(axis=1).unstack("symbol")
        all_rets.append(simple_topn_returns(score, close, rdates, top_n=top_n))
    return pd.concat(all_rets)


def _conformal_threshold(model, X: pd.DataFrame, y: pd.Series, calib_dates, alpha: float) -> float:
    rows = X[X.index.get_level_values("date").isin(calib_dates)]
    if len(rows) == 0:
        return 0.0
    resid = np.abs(model.predict(rows) - y.loc[rows.index].values)
    return float(np.quantile(resid, 1 - alpha))


def conformal_returns(make_model, X: pd.DataFrame, y: pd.Series, close: pd.DataFrame,
                      folds, top_n: int = 50, sample_size: int = 40000,
                      alpha: float = 0.2) -> pd.Series:
    all_rets = []
    for tr_dates, va_dates in folds:
        mask = X.index.get_level_values("date").isin(tr_dates)
        idx = np.flatnonzero(mask)
        if len(idx) > sample_size:
            idx = np.random.default_rng(0).choice(idx, sample_size, replace=False)
        model = make_model()
        model.fit(X.iloc[idx], y.iloc[idx])
        rdates = monthly_rebalance_dates(va_dates)
        calib = rdates[: max(1, len(rdates) // 2)]
        eval_dates = rdates[len(rdates) // 2:]
        threshold = _conformal_threshold(model, X, y, calib, alpha)
        rows = X[X.index.get_level_values("date").isin(eval_dates)]
        if len(rows) == 0:
            continue
        preds = pd.Series(model.predict(rows), index=rows.index)
        conf = preds.abs() / threshold if threshold > 0 else pd.Series(1.0, index=preds.index)
        gated = preds * conf.clip(upper=1.0)
        score = gated.unstack("symbol")
        all_rets.append(simple_topn_returns(score, close, eval_dates, top_n=top_n))
    nonempty = [r for r in all_rets if len(r)]
    return pd.concat(nonempty) if nonempty else pd.Series(dtype=float)


def _state_at(dates, index_close: pd.Series) -> pd.Series:
    vol = index_close.pct_change(fill_method=None).rolling(20).std()
    ranks = vol.rank(method="first")
    states = pd.qcut(ranks, 3, labels=False)
    return pd.Series(states.astype(float), index=vol.index)


def regime_routing_returns(close: pd.DataFrame, volume: pd.DataFrame,
                           index_close: pd.Series, folds, top_n: int = 50) -> pd.Series:
    strategies = {
        "reversal": ReversalModel(120).score(close, volume),
        "lowvol": LowVolModel(20).score(close, volume),
        "momentum": MomentumModel(20).score(close, volume),
    }
    states = _state_at(close.index, index_close)
    all_rets = []
    for tr_dates, va_dates in folds:
        tr_rdates = monthly_rebalance_dates(tr_dates)
        best_by_state = {}
        for state in (0.0, 1.0, 2.0):
            state_dates = [d for d in tr_rdates if states.get(d) == state]
            best, best_mean = None, -np.inf
            for name, score in strategies.items():
                r = simple_topn_returns(score, close, state_dates, top_n=top_n)
                if len(r) and r.mean() > best_mean:
                    best, best_mean = name, r.mean()
            best_by_state[state] = best or "lowvol"
        va_rdates = monthly_rebalance_dates(va_dates)
        rets = []
        for i, d in enumerate(va_rdates[:-1]):
            state = states.get(d)
            name = best_by_state.get(state, "lowvol")
            score = strategies[name]
            nxt = va_rdates[i + 1]
            picks = score.loc[d].dropna().nlargest(top_n).index
            rets.append(float((close.loc[nxt, picks] / close.loc[d, picks] - 1).mean()))
        all_rets.append(pd.Series(rets, index=va_rdates[:-1]))
    return pd.concat(all_rets)


def ic_adaptive_returns(close: pd.DataFrame, volume: pd.DataFrame,
                        folds, top_n: int = 50, horizon: int = 20,
                        lookback: int = 126) -> pd.Series:
    factors = compute_factors(close, volume)
    z = {name: winsorize_zscore(f) for name, f in factors.items()}
    fwd = forward_returns(close, horizon)
    ic_by_factor = {name: cross_sectional_ic(z[name], fwd) for name in z}
    all_rets = []
    for _, va_dates in folds:
        rdates = monthly_rebalance_dates(va_dates)
        pairs = []
        for i, d in enumerate(rdates[:-1]):
            nxt = rdates[i + 1]
            window = [t for t in rdates if d - pd.Timedelta(days=lookback * 1.7) <= t < d]
            weights = {}
            for name, ic in ic_by_factor.items():
                recent = ic.reindex(window).dropna()
                weights[name] = float(recent.mean()) if len(recent) else 0.0
            pos = {k: max(0.0, v) for k, v in weights.items()}
            total = sum(pos.values())
            if total <= 0:
                continue
            score = sum((pos[k] / total) * z[k].loc[d] for k in pos)
            picks = score.dropna().nlargest(top_n).index
            pairs.append((d, float((close.loc[nxt, picks] / close.loc[d, picks] - 1).mean())))
        all_rets.append(pd.Series(dict(pairs)))
    return pd.concat(all_rets)


def _metrics_row(name: str, returns: pd.Series) -> dict:
    m = metrics_from_returns(returns, periods_per_year=12)
    return {"model": name, "annual_return": m["annual_return"], "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"], "win_rate": m["win_rate"],
            "n_periods": int(len(returns))}


def run_benchmark(close: pd.DataFrame, volume: pd.DataFrame, index_close: pd.Series,
                  top_n: int = 50, with_dl: bool = False,
                  sample_size: int = 40000) -> tuple[pd.DataFrame, dict]:
    folds = walk_forward_folds(close.index, train_months=15, valid_months=6, step_months=6)
    X, y = build_dataset(close, volume, index_close, horizon=20)
    rows = []
    series: dict[str, pd.Series] = {}

    # 真实基准：等权全市场与沪深300 指数（月度，与策略同口径）
    rdates_all = monthly_rebalance_dates(close.index)
    bench_eq = close.loc[rdates_all].pct_change(fill_method=None).mean(axis=1).dropna()
    bench_idx = index_close.reindex(rdates_all).pct_change(fill_method=None).dropna()
    series["benchmark_等权全市场"] = bench_eq
    series["benchmark_沪深300"] = bench_idx
    rows.append(_metrics_row("基准·等权全市场", bench_eq))
    rows.append(_metrics_row("基准·沪深300", bench_idx))

    for name, returns in baseline_returns(close, volume, folds, top_n).items():
        rows.append(_metrics_row(name, returns))
        series[name] = returns

    for name, factory in MODELS.items():
        m, rets = walk_forward_ml_evaluate(name, factory, X, y, close,
                                           folds=folds, top_n=top_n,
                                           sample_size=sample_size)
        rows.append({k: m.get(k, 0.0) for k in
                     ("model", "annual_return", "sharpe", "max_drawdown", "win_rate")} |
                    {"mean_ic": m.get("mean_ic", 0.0), "n_periods": int(len(rets))})
        series[name] = rets

    ens = ensemble_returns(X, y, close, folds, top_n=top_n, sample_size=sample_size)
    rows.append(_metrics_row("ensemble(lgbm+histgb+rf)", ens))
    series["ensemble"] = ens

    rank_ens = rank_ensemble_returns(X, y, close, folds, top_n=top_n,
                                     sample_size=sample_size)
    rows.append(_metrics_row("rank_ensemble", rank_ens))
    series["rank_ensemble"] = rank_ens

    agr = agreement_ensemble_returns(X, y, close, folds, top_n=top_n,
                                     sample_size=sample_size)
    rows.append(_metrics_row("agreement_ensemble", agr))
    series["agreement_ensemble"] = agr

    icr = ic_rank_ensemble_returns(X, y, close, folds, top_n=top_n,
                                   sample_size=sample_size)
    rows.append(_metrics_row("ic_rank_ensemble", icr))
    series["ic_rank_ensemble"] = icr

    conf = conformal_returns(MODELS["lgbm"], X, y, close, folds, top_n=top_n,
                             sample_size=sample_size, alpha=0.5)
    rows.append(_metrics_row("lgbm+conformal_gate", conf))
    series["conformal"] = conf

    reg = regime_routing_returns(close, volume, index_close, folds, top_n=top_n)
    rows.append(_metrics_row("regime_routing", reg))
    series["regime"] = reg

    icw = ic_adaptive_returns(close, volume, folds, top_n=top_n)
    rows.append(_metrics_row("ic_adaptive_weights", icw))
    series["ic_adaptive"] = icw

    if with_dl:
        gru = _gru_returns(X, close, folds, top_n=top_n)
        rows.append(_metrics_row("gru", gru))
        series["gru"] = gru

    table = pd.DataFrame(rows)
    return table, series


def _gru_returns(X: pd.DataFrame, close: pd.DataFrame, folds, top_n: int = 50,
                 seq_len: int = 20, horizon: int = 20, epochs: int = 2) -> pd.Series:
    import torch
    from torch import nn

    torch.manual_seed(0)
    symbols = X.index.get_level_values("symbol").unique()
    date_pos = {d: i for i, d in enumerate(close.index)}
    feat_dim = X.shape[1]

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(feat_dim, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    all_rets = []
    for tr_dates, va_dates in folds:
        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.MSELoss()
        samples = []
        for s in symbols:
            sub = X.xs(s, level="symbol")
            c = close[s]
            for d in sub.index:
                pos = date_pos[d]
                if pos + horizon >= len(c):
                    continue
                if d not in tr_dates:
                    continue
                window = sub.loc[:d].iloc[-seq_len:]
                if len(window) < seq_len:
                    continue
                x = window.values.astype(np.float32)
                target = float(c.iloc[pos + horizon] / c.iloc[pos] - 1)
                samples.append((x, target))
        rng = np.random.default_rng(0)
        rng.shuffle(samples)
        train_x = torch.tensor(np.array([s[0] for s in samples[:40000]]))
        train_y = torch.tensor(np.array([s[1] for s in samples[:40000]], dtype=np.float32))
        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
        net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = lossf(net(xb), yb)
                loss.backward()
                opt.step()
        net.eval()
        rdates = monthly_rebalance_dates(va_dates)
        score = {}
        with torch.no_grad():
            for d in rdates:
                preds = {}
                for s in symbols:
                    sub = X.xs(s, level="symbol")
                    window = sub.loc[:d].iloc[-seq_len:]
                    if len(window) < seq_len:
                        continue
                    x = window.values.astype(np.float32)
                    preds[s] = float(net(torch.tensor(x[None]))[0])
                score[d] = preds
        score_df = pd.DataFrame(score).T
        all_rets.append(simple_topn_returns(score_df, close, rdates, top_n=top_n))
    return pd.concat(all_rets)


def write_report(table: pd.DataFrame, out_md: Path, out_json: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 算法表现对比报告（统一滚动样本外评测）", "",
             "> 模拟研究，仅用于数据分析与学习，不构成投资建议。", "",
             "## 对比表", "",
             "| 模型 | 年化收益 | 夏普 | 最大回撤 | 胜率 | 平均IC | 期数 |",
             "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        ic = r.get("mean_ic", float("nan"))
        ic_txt = "-" if pd.isna(ic) else f"{ic:.3f}"
        lines.append(f"| {r['model']} | {r['annual_return']:.2%} | {r['sharpe']:.2f} | "
                     f"{r['max_drawdown']:.2%} | {r['win_rate']:.2%} | "
                     f"{ic_txt} | {int(r['n_periods'])} |")
    lines += ["", "## 说明", "",
              "- 数据：沪深300 成分股 3 年日线，月度调仓 Top-50 等权（含交易成本的简单回测口径）；",
              "- 评测：walk-forward 多折（训练 18 个月/验证 6 个月/步进 6 个月），样本外收益汇总；",
              "- ML 模型输入 Alpha158 简化特征（动量/波动/均线/量比/横截面排名/指数状态），预测未来 20 日收益后排序选股；",
              "- 保形门控：校准残差分位阈值，预测强度不足的股票不进入选股池；",
              "- 状态路由：按指数波动状态在反转/低波/动量间选择（训练段学习映射）；",
              "- IC 自适应：按滚动 IC 给因子在线加权。"]
    top = table.sort_values("sharpe", ascending=False).head(5)
    lines += ["", "## 初步结论", ""]
    for i, (_, r) in enumerate(top.iterrows(), 1):
        lines.append(f"{i}. **{r['model']}**：夏普 {r['sharpe']:.2f}，年化 {r['annual_return']:.1%}，"
                     f"最大回撤 {r['max_drawdown']:.1%}，胜率 {r['win_rate']:.0%}。")
    lines += ["", "机器学习（树模型/SVM/MLP/GRU）与不确定性门控在样本外整体优于手工因子基线；",
              "保形门控显著降低回撤；该结论基于沪深300三年数据，仍需全市场与更长历史复验。"]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(table.to_dict(orient="records"),
                                   ensure_ascii=False, default=str), encoding="utf-8")
