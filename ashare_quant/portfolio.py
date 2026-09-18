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


DEFAULT_COSTS = {"commission": 0.00025, "stamp": 0.0005, "slippage": 0.001}
"""与 backtest/engine.py 的默认参数一致（佣金 0.025% + 印花税 0.05% + 滑点 0.1%）。"""


def _monthly_entries(entries: list[dict]) -> list[dict]:
    """按自然月取每月第一条决策作为当月调仓日（rebalance="M"）。

    为什么需要它（2026-09-16 实测）：回测与 ML 基准走 monthly_rebalance_dates（月频），
    而账户原先直接用相邻决策日分段 —— 决策是每天生成的，于是账户实际在"每日换仓"。
    实测 27 次 live 决策间隔 {1天:18, 2天:2, 3天:6}、单边换手率均值 59.4%，
    即账户跑的策略与"月度调仓 Top-50"根本不是同一个东西。config.rebalance 此前
    声明了却无人读取（死配置），这里让它真正生效。
    """
    out: list[dict] = []
    seen: set = set()
    for e in entries:
        key = pd.Timestamp(e["date"]).to_period("M")
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def equity_curve(close: pd.DataFrame, history: list[dict],
                 capital: float = 100000.0, costs: dict | None = None,
                 rebalance: str | None = None) -> pd.Series:
    """由决策历史计算日度账户收益序列（按决策日等权换仓）。

    costs: {"commission","stamp","slippage"} → 在每次换仓时扣交易成本；
        None = 不扣（**毛收益，仅供对比，不代表可实现收益**）。成本按单边换手计：
        卖出部分付 佣金+印花税+滑点，买入部分付 佣金+滑点；首次建仓全额买入。
    rebalance: "M" = 每月首个决策日才换仓（与回测/基准同口径）；
        None 或 "D" = 每次决策都换仓（改动前的旧行为）。
    """
    # 优先用正式决策（mode=live，样本外）；无正式记录时回退全部（参考用途）
    live = [e for e in history if e.get("mode") == "live"]
    entries = live if live else history
    if not entries:
        return pd.Series(dtype=float)
    if rebalance and str(rebalance).upper().startswith("M"):
        entries = _monthly_entries(entries)
    dates = close.index
    seg_returns = []
    prev_w: dict[str, float] = {}
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
        wmap = {s: float(w) for s, w in zip(syms, weights) if w > 0}
        # ---- 交易成本：按与上一期持仓的单边换手计 ----
        fee = 0.0
        if costs:
            c = float(costs.get("commission", 0.0))
            st = float(costs.get("stamp", 0.0))
            sl = float(costs.get("slippage", 0.0))
            if prev_w:
                allsym = set(prev_w) | set(wmap)
                sold = sum(max(0.0, prev_w.get(s, 0.0) - wmap.get(s, 0.0)) for s in allsym)
                bought = sum(max(0.0, wmap.get(s, 0.0) - prev_w.get(s, 0.0)) for s in allsym)
            else:
                sold, bought = 0.0, 1.0      # 首次建仓：全额买入
            fee = sold * (c + st + sl) + bought * (c + sl)
        prev_w = wmap
        prev = base.to_numpy(dtype=float)
        rets = []
        for t in seg:
            cur = close.loc[t, syms].to_numpy(dtype=float)
            day_ret = np.nan_to_num(cur / np.where(prev > 0, prev, np.nan) - 1, nan=0.0)
            rets.append(float((day_ret * weights).sum()))
            prev = np.where(np.isfinite(cur), cur, prev)
        if fee and rets:
            # 成本在换仓当日一次性扣掉（乘性，避免把小比例当线性叠加）
            rets[0] = (1 + rets[0]) * (1 - fee) - 1
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


def costs_from_config(cfg) -> dict:
    """从 Config 取交易成本三件套（缺失时用 engine.py 的默认值）。"""
    return {
        "commission": float(getattr(cfg, "commission", DEFAULT_COSTS["commission"])),
        "stamp": float(getattr(cfg, "stamp", DEFAULT_COSTS["stamp"])),
        "slippage": float(getattr(cfg, "slippage", DEFAULT_COSTS["slippage"])),
    }


def recompute_account(history: list[dict], close: pd.DataFrame,
                      capital: float = 100000.0, costs: dict | None = None,
                      rebalance: str | None = None) -> tuple[pd.Series, dict]:
    """按给定初始资金重算净值曲线与绩效指标（不写文件，仪表盘预览用）。"""
    returns = equity_curve(close, history, capital=capital, costs=costs,
                           rebalance=rebalance)
    if returns.empty:
        return pd.Series(dtype=float), {}
    equity = (1 + returns.fillna(0)).cumprod() * capital
    equity = _prepend_start_point(history, equity, capital)
    metrics = metrics_from_returns(returns, periods_per_year=252)
    return equity, metrics


def account_basis(history: list[dict], close: pd.DataFrame, date,
                  capital: float = 100000.0, costs: dict | None = None,
                  rebalance: str | None = None) -> float:
    """决策日时的累计净资产（自首个正式决策日跟踪的净值曲线取值）。

    实时估值若始终按「初始资金」作基准，会在每个决策日重置回初始值——
    例如 08-11 收盘刚生成决策时，本期收益恒为 0、总资产显示 10 万，
    与账户页累计净值（98,599）对不上。改用决策日累计净资产作基准后，
    实时总资产=决策日资产 × 现价/成本，收盘后即与账户页一致。

    costs/rebalance 必须与账户页、与 portfolio.update_portfolio 写盘时一致：
    否则实时估值会按另一套口径算出另一个基准（2026-09-16 实测：漏传这两个参数时
    实时页显示 106,313，而账户页已是 107,274，同一页面出现两个"总资产"）。
    """
    live = [e for e in history if e.get("mode") == "live"] or history
    equity, _ = recompute_account(live, close, capital, costs=costs,
                                  rebalance=rebalance)
    if equity.empty:
        return float(capital)
    d0 = pd.Timestamp(date)
    if d0 in equity.index:
        return float(equity.loc[d0])
    past = equity.index[equity.index <= d0]
    if len(past):
        return float(equity.loc[past[-1]])
    return float(equity.iloc[0])


def _prepend_start_point(history: list[dict], equity: pd.Series,
                         capital: float) -> pd.Series:
    """在净值曲线前补上首个决策日的建仓基准点（初始资金）。"""
    entries = [e for e in history if e.get("mode") == "live"] or history
    if not entries:
        return equity
    start = pd.Timestamp(entries[0]["date"])
    if start in equity.index:
        return equity
    if equity.empty:
        # 空曲线直接补建仓基准点；显式分支避免 pd.concat 空序列的弃用告警
        return pd.Series([float(capital)], index=[start])
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
    # 账户口径：扣交易成本 + 按 config.rebalance 的节奏换仓（默认 M=月频，与回测一致）。
    # 改动前这里既不扣成本也不读 rebalance，等于在模拟"每日全额换仓且零成本"，
    # 会把账面收益显著高估（2026-09-16 实测：成本一项就吃掉约 76% 的账面收益）。
    returns = equity_curve(close, history, float(cfg.initial_capital),
                           costs=costs_from_config(cfg),
                           rebalance=getattr(cfg, "rebalance", "M"))
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
