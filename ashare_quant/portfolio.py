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
        # 决策日 d0 收盘建仓，持有到下一决策日 d1 收盘（换仓前最后一刻），
        # 因此段 i 覆盖 (d0, d1]：含 d1 当天的收益；新仓位从 d1 之后开始。
        seg = dates[(dates > d0) & (dates <= d1)] if d1 is not None else dates[dates > d0]
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


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """年 × 月复合收益表（索引=年份，列=1..12，值为当月复合收益）。"""
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(r.index)
    t = pd.DataFrame({"y": idx.year, "m": idx.month, "r": r.to_numpy()})
    piv = t.pivot_table(index="y", columns="m", values="r",
                        aggfunc=lambda s: float((1 + s).prod() - 1))
    piv = piv.reindex(columns=range(1, 13))
    return piv


def recompute_account(history: list[dict], close: pd.DataFrame,
                      capital: float = 100000.0) -> tuple[pd.Series, dict]:
    """按给定初始资金重算净值曲线与绩效指标（不写文件，仪表盘预览用）。"""
    returns = equity_curve(close, history, capital=capital)
    if returns.empty:
        return pd.Series(dtype=float), {}
    equity = (1 + returns.fillna(0)).cumprod() * capital
    equity = _prepend_start_point(history, equity, capital)
    metrics = metrics_from_returns(returns, periods_per_year=252)
    return equity, metrics


def _prepend_start_point(history: list[dict], equity: pd.Series,
                         capital: float) -> pd.Series:
    """在净值曲线前补上首个决策日的建仓基准点（初始资金）。"""
    entries = [e for e in history if e.get("mode") == "live"] or history
    if not entries:
        return equity
    start = pd.Timestamp(entries[0]["date"])
    if start in equity.index:
        return equity
    return pd.concat([pd.Series([float(capital)], index=[start]), equity]).sort_index()


def build_trade_ledger(history: list[dict], close: pd.DataFrame,
                       capital: float = 100000.0) -> pd.DataFrame:
    """由决策历史推导交易台账（每次决策视为等权全换仓）。

    上一期持仓全部按当期收盘卖出（计入已实现盈亏），当期选股全部按当期
    收盘买入；与 equity_curve 的逐段收益模型口径一致。
    """
    rows: list[dict] = []
    prev: dict[str, dict] = {}  # symbol -> {"shares": float, "cost": float}
    for dec in history:
        d0 = pd.Timestamp(dec["date"])
        if d0 not in close.index:
            continue
        picks = dec.get("picks") or []
        cap = float(dec.get("capital") or dec.get("initial_capital") or capital)
        # 卖出上一期全部持仓
        for sym, pos in prev.items():
            if sym not in close.columns or pd.isna(close.loc[d0, sym]):
                continue
            price = float(close.loc[d0, sym])
            rows.append({
                "决策日": str(d0.date()), "代码": sym, "动作": "卖出",
                "数量": round(pos["shares"], 2), "价格": round(price, 3),
                "金额": round(pos["shares"] * price, 2),
                "实现盈亏": round(pos["shares"] * (price - pos["cost"]), 2),
            })
        if not picks:
            prev = {}
            continue
        syms = [p["symbol"] for p in picks]
        weights = np.asarray([float(p.get("weight", 0)) for p in picks], dtype=float)
        base = close.loc[d0, syms].astype(float)
        valid = base.notna().to_numpy() & (base.to_numpy() > 0)
        weights = weights * valid
        total = weights.sum()
        if total <= 0:
            prev = {}
            continue
        weights = weights / total
        current: dict[str, dict] = {}
        for i, sym in enumerate(syms):
            if not valid[i]:
                continue
            notional = cap * float(weights[i])
            price = float(base[sym])
            current[sym] = {"shares": notional / price, "cost": price}
            rows.append({
                "决策日": str(d0.date()), "代码": sym, "动作": "买入",
                "数量": round(current[sym]["shares"], 2),
                "价格": round(price, 3),
                "金额": round(notional, 2),
                "实现盈亏": 0.0,
            })
        prev = current
    cols = ["决策日", "代码", "动作", "数量", "价格", "金额", "实现盈亏"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


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
    equity = _prepend_start_point(history, equity, float(cfg.initial_capital))
    equity.to_csv(portfolio_dir / EQUITY_FILENAME, encoding="utf-8-sig")
    metrics = metrics_from_returns(returns, periods_per_year=252)
    live_decisions = len([e for e in history if e.get("mode") == "live"])
    summary = {
        "initial_capital": float(cfg.initial_capital),
        "total_asset": float(equity.iloc[-1]) if len(equity) else float(cfg.initial_capital),
        "total_return": float(equity.iloc[-1] / cfg.initial_capital - 1) if len(equity) else 0.0,
        "decisions": live_decisions or len(history),
        "backfilled": added,
        "as_of": str(close.index.max().date()),
        **{k: float(v) for k, v in metrics.items()},
    }
    (portfolio_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
