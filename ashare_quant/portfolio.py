"""决策组合历史追踪：记录每次决策 → 计算账户净值曲线与绩效指标。

模型：每个决策日按当日选股等权建仓，持有到下一决策日；用日线收盘价
逐日盯市值，形成持续更新的账户净值曲线（与同类模拟盘一致）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.metrics import metrics_from_returns
from .backtest.simple import monthly_rebalance_dates

HISTORY_FILENAME = "account_history.jsonl"
EQUITY_FILENAME = "account_equity.csv"
SUMMARY_FILENAME = "account_summary.json"


def load_history(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    entries = [json.loads(line) for line in
               Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(entries, key=lambda e: e.get("date", ""))


def append_decision(path: Path, decision: dict) -> None:
    """追加一条决策记录；同日期覆盖。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_history(path)
    date = str(pd.Timestamp(decision["date"]).date())
    entries = [e for e in entries if e.get("date") != date]
    entries.append(decision)
    entries.sort(key=lambda e: e.get("date", ""))
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries)
                    + "\n", encoding="utf-8")


def build_historical_decisions(close: pd.DataFrame, X: pd.DataFrame, models: dict,
                               cfg, history_path: Path, top_n: int = 50) -> int:
    """用已训练模型回填历史调仓日的决策（首次生成账户曲线需要）。"""
    from .ml.decision import decide

    history_path = Path(history_path)
    existing = {e["date"] for e in load_history(history_path)}
    rdates = [d for d in monthly_rebalance_dates(close.index)
              if str(d.date()) not in existing
              and d in close.index
              and d in X.index.get_level_values("date")]
    added = 0
    for d in rdates:
        try:
            picks = decide(models, X, close, d, top_n=top_n)
        except (KeyError, ValueError):
            continue
        append_decision(history_path, {
            "date": str(d.date()),
            "capital": float(getattr(cfg, "initial_capital", 100000.0)),
            "picks": picks.to_dict(orient="records"),
        })
        added += 1
    return added


def equity_curve(close: pd.DataFrame, history: list[dict],
                 capital: float = 100000.0) -> pd.Series:
    """由决策历史计算日度账户收益序列（按决策日等权换仓）。"""
    # 优先用正式决策（mode=live，样本外）；无正式记录时回退全部（参考用途）
    live = [e for e in history if e.get("mode") == "live"]
    entries = live if live else history
    if not entries:
        return pd.Series(dtype=float)
    dates = close.index
    seg_returns = []
    for i, dec in enumerate(entries):
        d0 = pd.Timestamp(dec["date"])
        if d0 not in close.index:
            continue
        d1 = pd.Timestamp(entries[i + 1]["date"]) if i + 1 < len(entries) else None
        seg = dates[(dates > d0) & (dates < d1)] if d1 is not None else dates[dates > d0]
        if len(seg) == 0:
            continue
        picks = dec.get("picks") or []
        if not picks:
            continue
        syms = [p["symbol"] for p in picks]
        weights = np.asarray([float(p.get("weight", 0)) for p in picks], dtype=float)
        base = close.loc[d0, syms].astype(float)
        valid = base.notna().to_numpy() & (base.to_numpy() > 0)
        weights = weights * valid
        total = weights.sum()
        if total <= 0:
            continue
        weights = weights / total
        prev = base.to_numpy(dtype=float)
        rets = []
        for t in seg:
            cur = close.loc[t, syms].to_numpy(dtype=float)
            day_ret = np.nan_to_num(cur / np.where(prev > 0, prev, np.nan) - 1, nan=0.0)
            rets.append(float((day_ret * weights).sum()))
            prev = np.where(np.isfinite(cur), cur, prev)
        seg_returns.append(pd.Series(rets, index=seg))
    if not seg_returns:
        return pd.Series(dtype=float)
    r = pd.concat(seg_returns).sort_index()
    r = r[~r.index.duplicated(keep="last")]
    return r


def update_portfolio(close: pd.DataFrame, cfg, portfolio_dir: Path,
                     decision: dict | None = None,
                     X: pd.DataFrame | None = None,
                     models: dict | None = None) -> dict:
    """追加当日决策（正式记录）→ 计算净值曲线与指标并保存。

    X/models 仅用于可选的样本内历史回填（参考用途，默认关闭）。
    """
    portfolio_dir = Path(portfolio_dir)
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    history_path = portfolio_dir / HISTORY_FILENAME
    added = 0
    if X is not None and models is not None:
        added = build_historical_decisions(close, X, models, cfg, history_path,
                                           top_n=cfg.top_n)
    if decision is not None:
        decision = {**decision, "mode": "live"}
        append_decision(history_path, decision)
    history = load_history(history_path)
    returns = equity_curve(close, history, float(cfg.initial_capital))
    equity = ((1 + returns.fillna(0)).cumprod() * float(cfg.initial_capital))
    equity.to_csv(portfolio_dir / EQUITY_FILENAME, encoding="utf-8-sig")
    metrics = metrics_from_returns(returns, periods_per_year=252)
    summary = {
        "initial_capital": float(cfg.initial_capital),
        "total_asset": float(equity.iloc[-1]) if len(equity) else float(cfg.initial_capital),
        "total_return": float(equity.iloc[-1] / cfg.initial_capital - 1) if len(equity) else 0.0,
        "decisions": len(history),
        "backfilled": added,
        "as_of": str(close.index.max().date()),
        **{k: float(v) for k, v in metrics.items()},
    }
    (portfolio_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
