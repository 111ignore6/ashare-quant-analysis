"""A股量化研究·模拟分析控制台（Streamlit 单页应用）。

运行： python -m streamlit run dashboard.py
功能：状态总览 / 数据下载与更新（后台任务+实时输出）/ 模拟盘 / 今日决策 / 算法对比 / 日志。
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from ashare_quant.account import account_snapshot
from ashare_quant.config import update_config_yaml
from ashare_quant.portfolio import (account_basis, build_trade_ledger,
                                    load_history, monthly_returns_table,
                                    recompute_account)
from ashare_quant.realtime import index_snapshot, snapshot

PROJECT = Path(__file__).parent
DISCLAIMER = "模拟研究，仅用于数据分析与学习，不构成投资建议。"
# 账户绩效指标（年化/夏普/回撤/胜率等）至少需要这么多交易日才有统计意义
MIN_METRIC_DAYS = 20

# 模型 / 算法 / 基准的中文名称映射
MODEL_NAMES = {
    "momentum": "动量(60日)",
    "reversal": "反转(60日)",
    "lowvol": "低波(60日)",
    "multifactor": "多因子",
    "rotation": "模型轮动",
    "momentum20": "动量(20日)",
    "reversal120": "反转(120日)",
    "lowvol20": "低波(20日)",
    "linear": "线性回归",
    "rf": "随机森林",
    "lgbm": "LightGBM",
    "histgb": "梯度提升(HistGB)",
    "svm": "支持向量机",
    "knn": "K近邻",
    "mlp": "神经网络(MLP)",
    "ensemble": "集成(LGBM+HistGB+RF)",
    "conformal": "LGBM+保形门控",
    "regime": "状态路由",
    "ic_adaptive": "IC自适应加权",
    "gru": "GRU深度模型",
    "benchmark_等权全市场": "基准·等权全市场",
    "benchmark_沪深300": "基准·沪深300",
    "ensemble(lgbm+histgb+rf)": "集成(LGBM+HistGB+RF)",
    "lgbm+conformal_gate": "LGBM+保形门控",
    "regime_routing": "状态路由",
    "ic_adaptive_weights": "IC自适应加权",
    "基准·等权全市场": "基准·等权全市场",
    "基准·沪深300": "基准·沪深300",
}

METRIC_NAMES = {
    "model": "模型",
    "annual_return": "年化收益",
    "sharpe": "夏普",
    "max_drawdown": "最大回撤",
    "win_rate": "胜率",
    "mean_ic": "平均IC",
    "n_periods": "期数",
}

PICK_NAMES = {"symbol": "代码", "score": "预期收益(20日)", "weight": "权重"}


def display_name(name: str) -> str:
    """列名 -> 中文展示名；已是中文或无法映射时原样返回。"""
    return MODEL_NAMES.get(str(name), str(name))


def format_metric(df: pd.DataFrame) -> pd.DataFrame:
    """指标表汉化并格式化（收益/回撤/胜率转百分比）。"""
    out = df.rename(columns=METRIC_NAMES)
    for col in ("年化收益", "最大回撤", "胜率"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "-")
    for col in ("夏普", "平均IC"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    if "模型" in out.columns:
        out["模型"] = out["模型"].map(display_name)
    return out


@st.cache_data
def load_csv(path: Path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data
def load_json(path: Path):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_decision(sim_dir: Path) -> dict | None:
    """读取最新决策：兼容多个输出目录（计划任务历史写 docs/simulation）。

    普通函数（非 st.cache_data）：内部直接读文件，避免嵌套缓存调用——
    Streamlit 的 cache_data 嵌套在跨刷新读取时可能抛 KeyError。
    """
    candidates = [
        sim_dir / "decision.json",
        PROJECT / "docs" / "simulation" / "decision.json",
        PROJECT / "docs" / "decision" / "decision.json",
    ]
    best = None
    for p in candidates:
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if d and (best is None or str(d.get("date", "")) > str(best.get("date", ""))):
            best = d
    return best


@st.cache_data(ttl=600)
def load_panel_close(data_dir: Path):
    """读取面板缓存中的收盘价矩阵（date × symbol）。"""
    p = data_dir / "panels" / "close.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(ttl=5)
def _cached_snapshot(symbols: tuple) -> pd.DataFrame:
    """同一页面运行内共享同一份快照（5 秒 TTL），避免多处显示不一致。"""
    return snapshot(list(symbols))


@st.cache_data(ttl=600)
def load_symbol(data_dir: Path, code: str):
    """读取单只股票本地日线（K线详情用，秒级）。"""
    p = data_dir / f"{code}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def account_figure(equity: pd.Series, data_dir: Path,
                   close_panel, capital: float) -> go.Figure:
    """账户净值曲线 + 真实基准（沪深300 / 等权全市场）。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity, mode="lines",
                             name="账户净值", line=dict(color="#2980b9")))
    start = equity.index[0]
    idx_path = data_dir / "sh000300.parquet"
    if idx_path.exists():
        idx_close = pd.read_parquet(idx_path)["close"]
        idx_sel = idx_close.loc[start:]
        if len(idx_sel) >= 2:
            bench = capital * idx_sel / idx_sel.iloc[0]
            fig.add_trace(go.Scatter(x=bench.index, y=bench, name="基准·沪深300",
                                     line=dict(dash="dash", color="#7f8c8d")))
    if close_panel is not None and start in close_panel.index:
        eq_ret = close_panel.loc[start:].mean(axis=1)
        if len(eq_ret) >= 2:
            bench2 = capital * eq_ret / eq_ret.iloc[0]
            fig.add_trace(go.Scatter(x=bench2.index, y=bench2,
                                     name="基准·等权全市场",
                                     line=dict(dash="dash", color="#95a5a6")))
    fig.update_layout(title="账户净值曲线（元，虚线为真实基准）", xaxis_title="日期",
                      yaxis_title="总资产（元）", hovermode="x unified")
    return fig


def format_pct_nan(v) -> str:
    """NaN 显示为 —，否则显示百分比。"""
    return "—" if pd.isna(v) else f"{v:+.2%}"


def render_realtime_valuation(decision: dict, data_dir: Path, key: str,
                              capital: float | None = None) -> None:
    """盘中实时估值（快照价）：开关开启才拉行情，避免每次刷新变慢。"""
    if decision is None or not decision.get("picks"):
        return
    if not st.toggle("盘中实时估值（快照价，秒级）", value=False, key=key):
        return
    try:
        initial = float(capital) if capital is not None \
            else float(decision.get("initial_capital", 100000.0))
        pick_syms = [p["symbol"] for p in decision["picks"]]
        snap = _cached_snapshot(tuple(pick_syms))
        if snap.empty:
            st.warning("未获取到实时行情（可能非交易时段或接口限流）。")
            return
        close_panel = load_panel_close(data_dir)
        # 以决策日累计净资产为基准：否则每个决策日实时总资产会重置回初始值，
        # 与账户页累计净值（98,599 而非 100,000）对不上
        history_path = data_dir / "portfolio" / "account_history.jsonl"
        hist_all = load_history(history_path) if history_path.exists() else []
        hist_live = [e for e in hist_all if e.get("mode") == "live"] or hist_all
        basis = account_basis(hist_live, close_panel, decision["date"], initial) \
            if close_panel is not None else initial
        d0 = pd.Timestamp(decision["date"])
        prices = snap.set_index("代码")["现价"].astype(float)
        fallback = close_panel.iloc[-1].reindex(pick_syms)
        if close_panel is not None and d0 in close_panel.index:
            fallback = fallback.fillna(close_panel.loc[d0, pick_syms])
        prices = prices.reindex(fallback.index).fillna(fallback)
        dec_view = {**decision, "initial_capital": basis}
        acc = account_snapshot(dec_view, close_panel, prices=prices)
        if acc is None:
            st.warning("无法按实时价估值（缺少决策日基准）。")
            return
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("实时总资产", f"{acc['total_asset']:,.0f} 元")
        r2.metric("实时总收益（累计）", f"{acc['total_asset'] / initial - 1:+.2%}")
        r3.metric("本期浮动盈亏", f"{acc['total_pnl']:+,.0f} 元")
        r4.metric("现金余额", f"{acc['cash']:,.0f} 元")
        st.caption(f"基准：决策日 {decision['date']} 累计净资产 {basis:,.0f} 元"
                   "（自首个决策日跟踪）× 快照现价逐只估值；实时总收益=总资产/初始资金-1，"
                   "与「账户」页累计口径一致。缺失行情用本地最新收盘价兜底，"
                   "收盘后现价=收盘价、本期浮动盈亏为 0。")
    except Exception as e:  # noqa: BLE001
        st.error(f"实时估值获取失败：{e}")


@st.cache_data(ttl=60)
def auto_update_status() -> str:
    """读取 Windows 计划任务状态（只读）。"""
    import subprocess
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-ScheduledTask -TaskName 'AshareQuantDaily').State"],
            capture_output=True, text=True, timeout=10)
        state = r.stdout.strip()
        return "开启" if state == "Ready" else ("关闭" if state == "Disabled" else state or "未注册")
    except Exception:  # noqa: BLE001
        return "未知"


def equity_figure(returns: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    equity = (1 + returns.fillna(0)).cumprod()
    for col in equity.columns:
        name = display_name(col)
        is_bench = str(col).startswith("基准") or str(col).startswith("benchmark_")
        dash = "dash" if is_bench else "solid"
        fig.add_trace(go.Scatter(x=equity.index, y=equity[col], mode="lines",
                                 name=name, line=dict(dash=dash)))
    fig.update_layout(title="策略净值曲线（模拟，含真实基准）", xaxis_title="日期",
                      yaxis_title="净值", legend_title="策略", hovermode="x unified")
    return fig


# ---------- 后台任务管理：在页面内直接跑数据下载/更新，实时回显输出 ----------

def _task_queue(key: str) -> queue.Queue:
    """任务队列存 session_state（跨页面刷新持久，后台线程与渲染共享）。"""
    st.session_state.setdefault("task_queues", {})
    return st.session_state["task_queues"].setdefault(key, queue.Queue())


def _task_worker(key: str, cmd: list[str], cwd: Path, q: queue.Queue) -> None:
    log_dir = cwd / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(log_dir / f"{key}.log", "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            log_f.write(line + "\n")
            log_f.flush()
            q.put(("line", line.rstrip()))
        proc.wait()
        q.put(("done", proc.returncode))
    except Exception as e:  # noqa: BLE001
        q.put(("line", f"启动失败：{e}"))
        q.put(("done", -1))
    finally:
        log_f.close()


def launch_task(key: str, cmd: list[str], cwd: Path) -> None:
    """启动一个后台任务（幂等：同一 key 运行中不重复启动）。"""
    state = st.session_state.setdefault("task_state", {}).setdefault(
        key, {"lines": [], "running": False, "code": None})
    if state["running"]:
        return
    q = _task_queue(key)
    while not q.empty():  # 清空上次遗留
        try:
            q.get_nowait()
        except queue.Empty:
            break
    state.update({"lines": [], "running": True, "code": None})
    threading.Thread(target=_task_worker, args=(key, cmd, cwd, q), daemon=True).start()


def render_task(key: str, title: str) -> bool:
    """渲染任务进度；返回是否仍在运行。"""
    state = st.session_state.setdefault("task_state", {}).setdefault(
        key, {"lines": [], "running": False, "code": None})
    q = st.session_state.get("task_queues", {}).get(key)
    if q is not None:
        while True:
            try:
                kind, payload = q.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                state["lines"].append(payload)
            elif kind == "done":
                state["running"] = False
                state["code"] = payload
    if state["running"]:
        with st.status(f"{title} 进行中…", expanded=True):
            tail = state["lines"][-30:]
            st.code("\n".join(tail) if tail else "等待输出…（长任务请耐心等待）")
        # 任务运行中每 5 秒自动刷新页面，实时显示进度
        st_autorefresh(interval=5000, key=f"task_refresh_{key}")
        return True
    if state["code"] == 0:
        st.success(f"{title} 完成")
    elif state["code"] is not None:
        st.error(f"{title} 失败（退出码 {state['code']}）")
    if state["lines"]:
        with st.expander("查看完整输出"):
            st.code("\n".join(state["lines"]))
    return False


st.set_page_config(page_title="A股量化研究·模拟分析控制台", layout="wide")
st.title("A股量化研究 · 模拟分析控制台")
st.caption(DISCLAIMER)

with st.sidebar:
    st.subheader("数据范围")
    data_root = st.selectbox("数据集", ["data/tencent", "data/all", "data/3y"], index=0)
    mode = "all" if data_root in ("data/tencent", "data/all") else "3y"
    sim_dir = PROJECT / ("docs/simulation-all" if mode == "all" else "docs/simulation")
    st.caption(f"数据目录：{PROJECT / data_root}")
    st.divider()
    cfg_path = PROJECT / "config.yaml"
    cfg_d = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) \
        if cfg_path.exists() else {}
    st.subheader("模拟参数")
    capital = st.number_input(
        "初始资金（元）", min_value=10000.0, max_value=100000000.0,
        value=float(cfg_d.get("initial_capital", 100000.0)),
        step=10000.0, format="%.0f")
    top_n = st.slider("每期选股数量", 10, 200,
                      int(cfg_d.get("top_n", 50)), step=5)
    use_sl = st.toggle("启用止损", value=cfg_d.get("stop_loss") is not None)
    sl = st.number_input("止损线（相对成本）", min_value=-0.50, max_value=0.0,
                         value=float(cfg_d.get("stop_loss", -0.15)),
                         step=0.01, format="%.2f", disabled=not use_sl)
    use_tp = st.toggle("启用止盈", value=cfg_d.get("take_profit") is not None)
    tp = st.number_input("止盈线（相对成本）", min_value=0.0, max_value=1.0,
                         value=float(cfg_d.get("take_profit", 0.30)),
                         step=0.01, format="%.2f", disabled=not use_tp)
    src_names = ["tencent", "akshare", "mootdx", "baostock"]
    try:
        src_idx = src_names.index(str(cfg_d.get("data_source", "tencent")))
    except ValueError:
        src_idx = 0
    data_source = st.selectbox("数据源", src_names, index=src_idx)
    if st.button("保存参数到配置"):
        update_config_yaml(
            PROJECT / "config.yaml",
            initial_capital=capital, top_n=top_n,
            stop_loss=sl if use_sl else None,
            take_profit=tp if use_tp else None,
            data_source=data_source)
        st.session_state["cfg_saved"] = True
        st.rerun()
    if st.session_state.pop("cfg_saved", False):
        st.success("已保存到 config.yaml，下次「每日更新」生效（账户页已按新资金预览）。")
    st.caption("提示：改资金只改变账户口径，不改变选股决策。")
    st.divider()
    st.caption("使用方式：在「总览」页点击按钮即可下载/更新数据，"
               "任务在后台运行、输出实时显示，刷新页面不中断。")
    st.divider()
    st.caption(DISCLAIMER)

tab_overview, tab_sim, tab_decision, tab_account, tab_realtime, tab_algo, tab_log, tab_data = st.tabs(
    ["总览", "模拟盘", "今日决策", "账户", "实时行情", "算法对比", "调整日志", "数据状态"])

data_dir = PROJECT / data_root
model_dir = PROJECT / "models" / mode

with tab_overview:
    st.subheader("系统状态与快速操作")
    manifest = load_json(data_dir / "manifest.json") if data_dir.exists() else None
    model_meta = load_json(model_dir / "meta.json")
    decision = load_decision(sim_dir)
    stocks = {k: v for k, v in (manifest or {}).items() if k != "sh000300"}
    idx_end = (manifest or {}).get("sh000300", {}).get("end")
    stale = [k for k, v in stocks.items()
             if v.get("end") and idx_end and v["end"] < idx_end]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("覆盖股票", f"{len(stocks)} 只")
    c2.metric("数据截止", idx_end or "无数据")
    c3.metric("落后股票", f"{len(stale)} 只")
    c4.metric("今日持仓", f"{len(decision['picks'])} 只" if decision else "—")

    close_panel = load_panel_close(data_dir)
    # 累计口径（与「账户」页一致）：自首个正式决策日跟踪的净值曲线
    history_path = data_dir / "portfolio" / "account_history.jsonl"
    hist_all = load_history(history_path) if history_path.exists() else []
    hist_live = [e for e in hist_all if e.get("mode") == "live"] or hist_all
    # 持仓明细/本期口径同样以决策日累计净资产为基准（避免重置回 10 万）
    basis = account_basis(hist_live, close_panel, decision["date"], float(capital)) \
        if decision is not None and close_panel is not None else float(capital)
    dec_view = {**decision, "initial_capital": basis} if decision is not None else None
    account = account_snapshot(dec_view, close_panel) \
        if dec_view and close_panel is not None else None
    equity_cum, _ = recompute_account(hist_live, close_panel, float(capital)) \
        if hist_live and close_panel is not None else (pd.Series(dtype=float), {})
    if account or not equity_cum.empty:
        st.subheader("模拟账户")
        if not equity_cum.empty:
            cum_asset = float(equity_cum.iloc[-1])
            cum_ret = cum_asset / float(capital) - 1
            cum_start = str(equity_cum.index[0].date())
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("初始资金", f"{float(capital):,.0f} 元")
            a2.metric("累计总资产", f"{cum_asset:,.0f} 元")
            a3.metric("累计收益率", f"{cum_ret:+.2%}")
            a4.metric("累计盈亏", f"{cum_asset - float(capital):+,.0f} 元")
            st.caption(f"累计口径：自 {cum_start} 首个决策日起按收盘价逐日盯市值，"
                       f"与「账户」页一致；截至 {equity_cum.index[-1].date()}。")
        else:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("初始资金", f"{account['initial']:,.0f} 元")
            a2.metric("本期总资产", f"{account['total_asset']:,.0f} 元")
            a3.metric("本期收益率", f"{account['total_return']:+.2%}")
            a4.metric("本期浮动盈亏", f"{account['total_pnl']:+,.0f} 元")
        if account:
            st.caption(f"本期口径：决策日 {decision['date']} 收盘建仓"
                       f"（基准 {basis:,.0f} 元 = 当日累计净资产），自决策日收益 "
                       f"{account['total_return']:+.2%}（刚决策当日为 0，次日开始体现）。")
        render_realtime_valuation(decision, data_dir, key="overview_rt",
                                  capital=float(capital))
    st.caption(f"每日自动更新：{auto_update_status()}（可在 start.bat 菜单 8 切换，默认开启）")

    if not manifest:
        st.warning("尚未下载数据。首次使用请点击下方「下载全市场数据」——"
                   "约 20-35 分钟，可断点续传（中断后重跑自动续传）。")
    elif stale:
        st.info(f"有 {len(stale)} 只股票数据落后（可能上次更新中断或停牌），"
                "点击「每日更新」自动补齐。")
    elif not model_meta:
        st.info("模型尚未训练，点击「每日更新」会自动训练（约 1 分钟）。")
    else:
        st.success("数据与模型就绪。每日收盘后点击「每日更新」："
                   "增量拉数据 → 生成报告 → 输出今日模拟持仓。")

    cmd_base = ["python", "-X", "utf8", "-u", "-m", "ashare_quant.cli"]
    b1, b2, b3 = st.columns(3)
    if b1.button("每日更新（推荐）", type="primary", width="stretch"):
        launch_task("daily", cmd_base + ["daily", "--config", str(PROJECT / "config.yaml"),
                                         "--data-root", str(data_dir), "--out-dir", str(sim_dir),
                                         "--model-dir", str(model_dir)], PROJECT)
    if b2.button("强制重算报告+决策", width="stretch"):
        launch_task("force", cmd_base + ["daily", "--config", str(PROJECT / "config.yaml"),
                                         "--data-root", str(data_dir), "--out-dir", str(sim_dir),
                                         "--model-dir", str(model_dir), "--force"], PROJECT)
    if b3.button("下载/更新全市场数据（首次 20-35 分钟）", width="stretch"):
        launch_task("fetch", cmd_base + ["fetch", "--config", str(PROJECT / "config.yaml"),
                                         "--universe", "all", "--data-root", str(data_dir),
                                         "--years", "3"], PROJECT)
    render_task("daily", "每日更新")
    render_task("force", "强制重算")
    render_task("fetch", "全市场数据下载")

    st.divider()
    st.subheader("今日模拟持仓（前 10）")
    if decision:
        picks = pd.DataFrame(decision["picks"]).rename(columns=PICK_NAMES).head(10)
        if "预期收益(20日)" in picks.columns:
            picks["预期收益(20日)"] = picks["预期收益(20日)"].map(
                lambda v: f"{v:.2%}" if pd.notna(v) else "-")
        if "权重" in picks.columns:
            picks["权重"] = picks["权重"].map(
                lambda v: f"{v:.1%}" if pd.notna(v) else "-")
        st.dataframe(picks, width="stretch")
        st.caption(f"决策日期 {decision['date']}，模型："
                   f"{'、'.join(display_name(m) for m in decision['models'])}。"
                   "预期收益为多模型预测的未来 20 个交易日收益均值。")
    else:
        st.info("暂无决策结果，运行「每日更新」后生成。")

with tab_sim:
    st.subheader("模拟盘对比（月度调仓 Top-50，含交易成本）")
    st.caption("虚线为真实市场基准：等权全市场与沪深300 指数买入持有。")
    st.caption("说明：此处为早期 5 个基础策略（动量/反转/低波/多因子/轮动）的月度调仓回测，"
               "收益按调仓月标记（如 07-01 段已含最新交易日数据）；"
               "当前每日决策使用的是 6 个 ML 模型，其样本外表现请见「算法对比」页。")
    risk_cfg = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8")) \
        if (PROJECT / "config.yaml").exists() else None
    if risk_cfg and (risk_cfg.get("stop_loss") is not None or risk_cfg.get("take_profit") is not None):
        sl = risk_cfg.get("stop_loss")
        tp = risk_cfg.get("take_profit")
        st.caption(f"仓位风控已启用：止损 {sl:+.0%}、止盈 {tp:+.0%}"
                   f"（config.yaml 可调，None 关闭）。")
    returns = load_csv(sim_dir / "model_returns.csv")
    if returns is None:
        st.info("未找到模拟盘结果，在「总览」运行「每日更新」或模拟盘命令后生成。")
    else:
        st.plotly_chart(equity_figure(returns), width="stretch")
        st.subheader("累计总收益（近三年模拟）")
        cum = (1 + returns.fillna(0)).prod() - 1
        # 只显示策略（基准线已在净值曲线中，避免把大盘涨幅当策略收益）
        strategy_cum = cum[[c for c in cum.index if not str(c).startswith("基准")]]
        cols = st.columns(len(strategy_cum))
        for col, (name, v) in zip(cols, strategy_cum.items()):
            col.metric(display_name(name), f"{v:+.2%}")
        sim_json = load_json(sim_dir / "simulation.json")
        if sim_json and "summary" in sim_json:
            st.dataframe(format_metric(pd.DataFrame(sim_json["summary"])), width="stretch")

with tab_decision:
    st.subheader("今日模拟投资决策")
    if decision is None:
        st.info("未找到决策结果，在「总览」运行「每日更新」后生成。")
    else:
        model_names = "、".join(display_name(m) for m in decision["models"])
        st.write(f"决策日期：{decision['date']}　持仓 {len(decision['picks'])} 只　"
                 f"模型：{model_names}")
        picks = pd.DataFrame(decision["picks"]).rename(columns=PICK_NAMES)
        if "预期收益(20日)" in picks.columns:
            picks["预期收益(20日)"] = picks["预期收益(20日)"].map(
                lambda v: f"{v:.2%}" if pd.notna(v) else "-")
        if "权重" in picks.columns:
            picks["权重"] = picks["权重"].map(
                lambda v: f"{v:.1%}" if pd.notna(v) else "-")
        st.dataframe(picks, width="stretch")
        st.caption("预期收益为多模型预测的未来 20 个交易日收益均值，模拟研究仅供学习。")
        if account:
            st.subheader("账户持仓明细")
            pos = account["rows"].copy()
            pos["权重"] = pos["权重"].map(lambda v: f"{v:.1%}")
            pos["投入金额"] = pos["投入金额"].map(lambda v: f"{v:,.0f}")
            pos["股数"] = pos["股数"].map(lambda v: f"{v:,.0f}")
            pos["成本价"] = pos["成本价"].map(lambda v: f"{v:.2f}")
            pos["现价"] = pos["现价"].map(lambda v: f"{v:.2f}")
            pos["市值"] = pos["市值"].map(lambda v: f"{v:,.0f}")
            pos["浮动盈亏"] = pos["浮动盈亏"].map(lambda v: f"{v:+,.0f}")
            pos["盈亏率"] = pos["盈亏率"].map(lambda v: f"{v:+.2%}")
            st.dataframe(pos, width="stretch")
        with st.expander("模型预测明细（为什么选这些股票）"):
            st.caption("每只股票在 LGBM / 梯度提升 / 随机森林 / SVM / KNN / "
                       "线性回归六个模型下的未来 20 日预期收益，"
                       "最终得分为多模型均值经置信度加权。")
            detail = pd.DataFrame(decision["picks"]).copy()
            if "model_scores" in detail.columns and detail["model_scores"].notna().any():
                scores = pd.json_normalize(detail["model_scores"].dropna().tolist())
                detail = pd.concat([detail[["symbol", "score"]], scores], axis=1)
                detail = detail.rename(columns={"symbol": "代码", "score": "加权得分"})
                st.dataframe(detail, width="stretch")
            else:
                st.info("当前决策文件缺少模型明细，重新运行 daily 后自动生成。")

        st.subheader("相对上一决策日变化")
        history_path = data_dir / "portfolio" / "account_history.jsonl"
        hist_all = load_history(history_path) if history_path.exists() else []
        hist_live = [e for e in hist_all if e.get("mode") == "live"] or hist_all
        if len(hist_live) >= 2:
            cur_dec, prev_dec = hist_live[-1], hist_live[-2]
            cur_date, prev_date = cur_dec.get("date", "?"), prev_dec.get("date", "?")
            cur_map = {p["symbol"]: p for p in cur_dec.get("picks", [])}
            prev_map = {p["symbol"]: p for p in prev_dec.get("picks", [])}
            cur_syms, prev_syms = set(cur_map), set(prev_map)
            diff_rows = []
            for s in sorted(cur_syms - prev_syms):
                p = cur_map[s]
                diff_rows.append({"代码": s, "变化": "🆕 新增", "当前权重": p.get("weight"),
                                  "预期收益(20日)": p.get("score"), "_score": p.get("score")})
            for s in sorted(prev_syms - cur_syms):
                p = prev_map[s]
                diff_rows.append({"代码": s, "变化": "➖ 卖出", "当前权重": None,
                                  "预期收益(20日)": p.get("score"), "_score": p.get("score")})
            for s in sorted(cur_syms & prev_syms):
                p = cur_map[s]
                diff_rows.append({"代码": s, "变化": "✅ 持有", "当前权重": p.get("weight"),
                                  "预期收益(20日)": p.get("score"), "_score": p.get("score")})
            order = {"🆕 新增": 0, "➖ 卖出": 1, "✅ 持有": 2}
            diff_rows.sort(key=lambda r: (order[r["变化"]], -((r["_score"] or 0))))
            for row in diff_rows:
                row.pop("_score", None)
            m1, m2, m3 = st.columns(3)
            m1.metric("🆕 新增", f"{len(cur_syms - prev_syms)} 只")
            m2.metric("➖ 卖出", f"{len(prev_syms - cur_syms)} 只")
            m3.metric("✅ 持有", f"{len(cur_syms & prev_syms)} 只")
            df_diff = pd.DataFrame(diff_rows, columns=["代码", "变化", "当前权重", "预期收益(20日)"])
            if "当前权重" in df_diff.columns:
                df_diff["当前权重"] = df_diff["当前权重"].map(
                    lambda v: f"{v:.1%}" if pd.notna(v) else "-")
            if "预期收益(20日)" in df_diff.columns:
                df_diff["预期收益(20日)"] = df_diff["预期收益(20日)"].map(
                    lambda v: f"{v:.2%}" if pd.notna(v) else "-")
            st.caption(f"对比 {prev_date} → {cur_date}：新增=本期新买入（按预期收益降序），"
                       "卖出=上期持有本期剔除，持有=两期都在。")
            st.dataframe(df_diff, width="stretch")
        else:
            st.caption("暂无上一决策日对比（账户刚开始记录）。")

        st.subheader("个股K线（本地数据，最近 120 个交易日）")
        sel = st.selectbox("选择个股", [p["symbol"] for p in decision["picks"]])
        sym_df = load_symbol(data_dir, sel)
        if sym_df is None or sym_df.empty:
            st.info("本地无该股K线数据（可能为新上市或数据缺失）。")
        else:
            tail = sym_df.tail(120)
            kfig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 vertical_spacing=0.03, row_heights=[0.75, 0.25])
            kfig.add_trace(go.Candlestick(
                x=tail.index, open=tail["open"], high=tail["high"],
                low=tail["low"], close=tail["close"], name=sel), row=1, col=1)
            for n in (5, 20, 60):
                kfig.add_trace(go.Scatter(x=tail.index,
                                          y=tail["close"].rolling(n).mean(),
                                          name=f"MA{n}", line=dict(width=1)),
                               row=1, col=1)
            if "volume" in tail.columns:
                kfig.add_trace(go.Bar(x=tail.index, y=tail["volume"],
                                      name="成交量", marker_color="#bdc3c7"),
                               row=2, col=1)
            kfig.update_layout(title=f"{sel} K线（最近 {len(tail)} 个交易日）",
                               height=560, xaxis_rangeslider_visible=False)
            st.plotly_chart(kfig, width="stretch")

with tab_account:
    st.subheader("模拟账户净值（自首个正式决策日跟踪）")
    st.caption("账户自首个正式决策日（样本外）开始逐日盯市值；"
               "决策每日收盘后生成，收益随每日更新持续累积。"
               "曲线为收盘价口径（每日更新后刷新）；盘中实时估值见下方开关。"
               "历史策略表现请参考「模拟盘」页。")
    render_realtime_valuation(decision, data_dir, key="account_rt",
                              capital=float(capital))
    equity_csv = load_csv(data_dir / "portfolio" / "account_equity.csv")
    acc_summary = load_json(data_dir / "portfolio" / "account_summary.json")
    history_path = data_dir / "portfolio" / "account_history.jsonl"
    history = load_history(history_path) if history_path.exists() else []
    close_panel = load_panel_close(data_dir)
    if equity_csv is None or equity_csv.empty:
        if acc_summary:
            st.info("首个正式决策已生成，账户净值曲线将于下一交易日（每日更新后）开始记录。")
            a1, a2 = st.columns(2)
            a1.metric("总资产", f"{acc_summary.get('total_asset', 0.0):,.0f} 元")
            a2.metric("总收益率", f"{acc_summary.get('total_return', 0.0):+.2%}")
            st.info(f"账户运行���足 {MIN_METRIC_DAYS} 个交易日，"
                    "年化/夏普/回撤等绩效指标暂无统计意义，曲线成形后自动展示。")
            st.caption(f"已记录 {acc_summary.get('decisions', 0)} 次决策，"
                       f"数据截至 {acc_summary.get('as_of', '-')}。")
        else:
            st.info("暂无账户曲线，运行「每日更新」生成首个正式决策后开始记录。")
    else:
        view_capital = float(capital)
        hist_view = [e for e in history if e.get("mode") == "live"] or history
        equity, metrics = recompute_account(hist_view, close_panel, view_capital) \
            if close_panel is not None and hist_view else (pd.Series(dtype=float), {})
        if equity.empty:
            # 回退到已保存的 CSV 与 summary（如缺少面板/历史）
            equity = equity_csv[equity_csv.columns[0]].astype(float)
            metrics = acc_summary or {}
            view_capital = float(acc_summary.get("initial_capital", view_capital)) \
                if acc_summary else view_capital
        fig = account_figure(equity, data_dir, close_panel, view_capital)
        st.plotly_chart(fig, width="stretch")
        if len(equity) < 5:
            st.info(f"账户刚开始记录（当前仅 {max(1, len(equity) - 1)} 个交易日收益），"
                    "曲线会随每日更新逐步成形；"
                    "想看完整历史策略表现，请切换到「模拟盘」页。")
        total_asset = float(equity.iloc[-1])
        total_return = total_asset / view_capital - 1 if view_capital > 0 else 0.0
        enough = len(equity) >= MIN_METRIC_DAYS
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("总资产", f"{total_asset:,.0f} 元")
        a2.metric("总收益率", f"{total_return:+.2%}")
        a3.metric("年化收益", format_pct_nan(metrics.get("annual_return")) if enough else "—")
        a4.metric("夏普", f"{metrics.get('sharpe', 0.0):.2f}" if enough else "—")
        a5.metric("最大回撤", format_pct_nan(metrics.get("max_drawdown")) if enough else "—")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("胜率", format_pct_nan(metrics.get("win_rate")) if enough else "—")
        plr = metrics.get("profit_loss_ratio", float("nan"))
        b2.metric("盈亏比", f"{plr:.2f}" if enough and not pd.isna(plr) else "—")
        b3.metric("年化波动", format_pct_nan(metrics.get("annual_vol")) if enough else "—")
        calmar = metrics.get("calmar", float("nan"))
        b4.metric("Calmar", f"{calmar:.2f}" if enough and not pd.isna(calmar) else "—")
        if not enough:
            st.info(f"账户运行仅 {len(equity)} 个净值点（不足 {MIN_METRIC_DAYS} 天），"
                    "年化/夏普/回撤/胜率等指标暂无统计意义，曲线成形后自动展示。")
        decisions_n = (acc_summary or {}).get("decisions", len(hist_view))
        as_of = (acc_summary or {}).get("as_of", str(equity.index[-1].date()))
        st.caption(f"已记录 {decisions_n} 次决策，数据截至 {as_of}；"
                   f"账户按侧边栏资金 {view_capital:,.0f} 元预览，"
                   "保存参数后每日更新沿用新口径。")

        # 账户 vs 基准同期收益对比
        if len(equity) >= 2 and close_panel is not None:
            start, end = equity.index[0], equity.index[-1]
            bench_ret = {}
            idx_path = data_dir / "sh000300.parquet"
            if idx_path.exists():
                idx_close = pd.read_parquet(idx_path)["close"]
                idx_sel = idx_close.loc[start:end]
                if len(idx_sel) >= 2:
                    bench_ret["基准·沪深300"] = idx_sel.iloc[-1] / idx_sel.iloc[0] - 1
            if start in close_panel.index:
                eq_ret = close_panel.loc[start:end].mean(axis=1)
                if len(eq_ret) >= 2:
                    bench_ret["基准·等权全市场"] = eq_ret.iloc[-1] / eq_ret.iloc[0] - 1
            st.subheader("同期收益对比（账户 vs 真实市场）")
            cc = st.columns(1 + len(bench_ret))
            cc[0].metric("模拟账户", f"{total_return:+.2%}")
            for col, (name, v) in zip(cc[1:], bench_ret.items()):
                col.metric(name, f"{v:+.2%}")

        # 月度收益热力图
        returns_v = equity.pct_change(fill_method=None).dropna()
        monthly = monthly_returns_table(returns_v)
        if not monthly.empty:
            st.subheader("月度收益热力图")
            mvals = monthly.values
            text = np.vectorize(lambda v: f"{v:.1%}" if pd.notna(v) else "")(mvals)
            hfig = go.Figure(go.Heatmap(
                z=mvals * 100, x=[f"{m}月" for m in monthly.columns],
                y=[str(y) for y in monthly.index], colorscale="RdYlGn", zmid=0,
                text=text, texttemplate="%{text}",
                hovertemplate="%{y}年 %{x}: %{z:.2f}%<extra></extra>"))
            hfig.update_layout(title="月度收益（%）", height=max(220, 45 * len(monthly.index)),
                               yaxis_title="年份")
            st.plotly_chart(hfig, width="stretch")

        # 交易台账
        if close_panel is not None and hist_view:
            ledger = build_trade_ledger(hist_view, close_panel, view_capital)
            if not ledger.empty:
                st.subheader("交易台账（每次决策视为等权全换仓）")
                shown = ledger.copy()
                for coln in ("数量", "价格", "金额", "实现盈亏"):
                    shown[coln] = shown[coln].map(lambda v: f"{v:,.2f}")
                st.dataframe(shown, width="stretch")
                st.download_button("下载交易台账 CSV", ledger.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="trade_ledger.csv", mime="text/csv")

        csv_data = equity.to_csv().encode("utf-8-sig")
        st.download_button("下载账户净值 CSV", data=csv_data,
                           file_name="account_equity.csv", mime="text/csv")

with tab_realtime:
    st.subheader("大盘速览")
    try:
        idx_df = index_snapshot()
        if idx_df.empty:
            st.warning("未获取到指数行情（可能非交易时段或接口限流）。")
        else:
            icols = st.columns(len(idx_df))
            for icol, (_, row) in zip(icols, idx_df.iterrows()):
                icol.metric(row["名称"], f"{row['现价']:,.2f}", f"{row['涨跌幅']:+.2%}")
    except Exception as e:  # noqa: BLE001
        st.warning(f"指数行情获取失败：{e}")
    st.divider()
    st.subheader("实时行情（准实时快照，秒级延迟）")
    st.caption("免费行情源为快照级（延迟数秒），非交易所级 tick 数据；"
               "仅供盘中观察与持仓跟踪，不改变月度调仓决策逻辑。")
    auto = st.toggle("自动刷新（每 10 秒）", value=False)
    if auto:
        st_autorefresh(interval=10_000, key="realtime_refresh")
    if decision is None or not decision.get("picks"):
        st.info("暂无持仓，先在「总览」运行「每日更新」生成决策。")
    else:
        symbols = [p["symbol"] for p in decision["picks"]]
        try:
            snap = _cached_snapshot(tuple(symbols))
            if snap.empty:
                st.warning("未获取到实时行情（可能非交易时段或接口限流），"
                           "可切换数据源重试。")
            else:
                close_panel = load_panel_close(data_dir)
                kline_lookup = snap.set_index("代码")[
                    ["现价", "今开", "最高", "最低", "成交量(手)"]].to_dict("index") \
                    if {"现价", "今开", "最高", "最低", "成交量(手)"} <= set(snap.columns) else {}
                acc_realtime = None
                d0 = pd.Timestamp(decision["date"])
                if close_panel is not None and d0 in close_panel.index:
                    # 实时估值：快照现价优先，缺失（停牌等）依次用本地最新收盘价、
                    # 决策日收盘价兜底（保持估值连续）
                    prices = snap.set_index("代码")["现价"].astype(float)
                    pick_syms = [p["symbol"] for p in decision["picks"]]
                    fallback = close_panel.iloc[-1].reindex(pick_syms)
                    fallback = fallback.fillna(close_panel.loc[d0, pick_syms])
                    prices = prices.reindex(fallback.index).fillna(fallback)
                    history_path = data_dir / "portfolio" / "account_history.jsonl"
                    hist_all = load_history(history_path) if history_path.exists() else []
                    hist_live = [e for e in hist_all if e.get("mode") == "live"] or hist_all
                    basis = account_basis(hist_live, close_panel,
                                          decision["date"], float(capital))
                    try:
                        acc_realtime = account_snapshot(
                            {**decision, "initial_capital": basis},
                            close_panel, prices=prices)
                    except Exception:  # noqa: BLE001
                        acc_realtime = None
                    if acc_realtime is not None:
                        st.subheader("实时账户估值（按快照现价）")
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("实时总资产", f"{acc_realtime['total_asset']:,.0f} 元")
                        r2.metric("实时总收益（累计）",
                                  f"{acc_realtime['total_asset'] / float(capital) - 1:+.2%}")
                        r3.metric("本期浮动盈亏", f"{acc_realtime['total_pnl']:+,.0f} 元")
                        r4.metric("现金余额", f"{acc_realtime['cash']:,.0f} 元")
                        st.caption(f"基准：决策日 {decision['date']} 累计净资产 "
                                   f"{basis:,.0f} 元（自首个决策日跟踪）× 快照现价；"
                                   "实时总收益=总资产/初始资金-1（累计口径，与账户页一致）。"
                                   "「自决策日涨跌」为现价相对决策日收盘的变化，"
                                   "「本期浮动盈亏」为现价相对决策日成本。")
                        # 每只股票实时收益列（数值版，先于下方格式化）
                        rows_r = acc_realtime["rows"].rename(columns={
                            "现价": "实时价", "市值": "持仓市值",
                            "浮动盈亏": "实时盈亏(元)", "盈亏率": "实时盈亏率",
                        })
                        snap = snap.merge(
                            rows_r[["代码", "成本价", "实时价", "持仓市值",
                                    "实时盈亏(元)", "实时盈亏率"]],
                            on="代码", how="left")
                if close_panel is not None and d0 in close_panel.index:
                    base = close_panel.loc[pd.Timestamp(decision["date"]), snap["代码"]]
                    snap["决策日收盘"] = base.to_numpy()
                    snap["自决策日涨跌"] = snap["现价"] / snap["决策日收盘"] - 1
                    snap["自决策日涨跌"] = snap["自决策日涨跌"].map(
                        lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
                    snap["决策日收盘"] = snap["决策日收盘"].map(
                        lambda v: f"{v:.2f}" if pd.notna(v) else "-")
                if acc_realtime is not None:
                    for col in ("成本价", "实时价"):
                        if col in snap.columns:
                            snap[col] = snap[col].map(
                                lambda v: f"{v:.2f}" if pd.notna(v) else "-")
                    if "持仓市值" in snap.columns:
                        snap["持仓市值"] = snap["持仓市值"].map(
                            lambda v: f"{v:,.0f}" if pd.notna(v) else "-")
                    if "实时盈亏(元)" in snap.columns:
                        snap["实时盈亏(元)"] = snap["实时盈亏(元)"].map(
                            lambda v: f"{v:+,.0f}" if pd.notna(v) else "-")
                    if "实时盈亏率" in snap.columns:
                        snap["实时盈亏率"] = snap["实时盈亏率"].map(
                            lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
                snap["涨跌幅"] = snap["涨跌幅"].map(lambda v: f"{v:+.2%}")
                for col in ("现价", "今开", "最高", "最低", "昨收"):
                    if col in snap.columns:
                        snap[col] = snap[col].map(
                            lambda v: f"{v:.2f}" if pd.notna(v) else "-")
                st.dataframe(snap, width="stretch")
                st.caption(f"决策日期 {decision['date']}，共 {len(snap)} 只；"
                           "「自决策日涨跌」为现价相对决策日收盘的变化，"
                           "「实时盈亏率」为现价相对持仓成本。")
                # 动态K线：本地日线 + 实时快照拼接当日 bar，随 10s 自动刷新动态更新
                st.subheader("动态K线（自动刷新时实时更新）")
                ksym = st.selectbox("选择个股",
                                    [p["symbol"] for p in decision["picks"]],
                                    key="realtime_kline_symbol")
                kdf = load_symbol(data_dir, ksym)
                if kdf is None or kdf.empty:
                    st.info("本地无该股K线数据。")
                else:
                    tail = kdf.tail(60).copy()
                    q = kline_lookup.get(ksym)
                    today = pd.Timestamp.today().normalize()
                    if q is not None and tail.index.max().date() < today.date():
                        now = q.get("现价")
                        if now not in (None, "-") and float(now) > 0:
                            now = float(now)
                            o = float(q["今开"]) if q.get("今开") not in (None, "-") else now
                            hi = float(q["最高"]) if q.get("最高") not in (None, "-") else now
                            lo = float(q["最低"]) if q.get("最低") not in (None, "-") else now
                            vol = float(q.get("成交量(手)") or 0) * 100
                            tail.loc[today] = {
                                "open": o, "high": max(hi, now), "low": min(lo, now),
                                "close": now, "volume": vol,
                            }
                    tail = tail.sort_index()
                    kfig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                         vertical_spacing=0.03, row_heights=[0.75, 0.25])
                    kfig.add_trace(go.Candlestick(
                        x=tail.index, open=tail["open"], high=tail["high"],
                        low=tail["low"], close=tail["close"], name=ksym), row=1, col=1)
                    for n in (5, 20, 60):
                        kfig.add_trace(go.Scatter(x=tail.index,
                                                  y=tail["close"].rolling(n).mean(),
                                                  name=f"MA{n}", line=dict(width=1)),
                                       row=1, col=1)
                    if "volume" in tail.columns:
                        kfig.add_trace(go.Bar(x=tail.index, y=tail["volume"],
                                              name="成交量", marker_color="#bdc3c7"),
                                       row=2, col=1)
                    kfig.update_layout(title=f"{ksym} 动态K线（最近 {len(tail)} 个交易日）",
                                       height=520, xaxis_rangeslider_visible=False)
                    st.plotly_chart(kfig, width="stretch")
                    st.caption("最后一根为实时快照拼接的当日 bar（现价/高低随刷新更新）；"
                               "开启上方「自动刷新」后每 10 秒动态变化。")
        except Exception as e:  # noqa: BLE001
            st.error(f"实时行情获取失败：{e}")

with tab_algo:
    st.subheader("算法表现对比（样本外夏普）")
    bench_stem = "algorithm-benchmark-all" if mode == "all" else "algorithm-benchmark"
    bench = load_json(PROJECT / "docs/research" / f"{bench_stem}.json")
    if bench is None:
        st.info("未找到算法对比结果，运行 benchmark 命令后生成（全市场约 6 分钟）。")
    else:
        df = pd.DataFrame(bench).sort_values("sharpe", ascending=False)
        fig = go.Figure(go.Bar(x=df["model"], y=df["sharpe"],
                               marker_color=["#c0392b" if v >= 2 else "#2980b9"
                                             for v in df["sharpe"]]))
        fig.update_layout(title="各算法样本外夏普", xaxis_title="模型", yaxis_title="夏普",
                          xaxis_tickangle=-30)
        fig.data[0].x = [display_name(m) for m in df["model"]]
        st.plotly_chart(fig, width="stretch")
        st.dataframe(format_metric(df), width="stretch")
        returns = load_csv(PROJECT / "docs/research" / f"{bench_stem}.returns.csv")
        if returns is not None:
            st.subheader("各算法样本外净值曲线（含真实基准）")
            st.caption("虚线为真实市场基准；曲线为月度调仓 Top-50 等权的样本外净值。")
            st.plotly_chart(equity_figure(returns), width="stretch")
        else:
            st.info("暂无收益曲线数据，重跑 benchmark 命令后自动生成。")

with tab_log:
    st.subheader("反馈调整日志")
    LOG_NAMES = {"date": "日期", "trigger": "触发", "action": "动作",
                 "before": "调整前权重", "after": "调整后权重", "effect": "效果"}
    log_path = sim_dir / "adjustments.jsonl"
    if not log_path.exists():
        st.info("暂无调整日志。")
    else:
        entries = [json.loads(line) for line in
                   log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        st.dataframe(pd.DataFrame(entries).rename(columns=LOG_NAMES), width="stretch")

with tab_data:
    st.subheader("数据状态")
    if data_dir.exists():
        stats_path = data_dir / "update_stats.json"
        if stats_path.exists():
            stats = load_json(stats_path)
            st.subheader("最近一次每日更新")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("运行时间", str(stats.get("last_run", "-")))
            s2.metric("数据源", str(stats.get("source", "-")))
            s3.metric("更新股票", f"{stats.get('updated', 0)} 只")
            s4.metric("总耗时", f"{stats.get('total_sec', '-')} 秒")
            st.caption(
                f"阶段耗时：数据拉取 {stats.get('phase1_sec', '-')}s / "
                f"面板构建 {stats.get('phase2_sec', '-')}s / "
                f"报告+决策 {stats.get('phase3_sec', '-')}s"
                + (f"（其中报告 {stats.get('report_sec')}s、决策 {stats.get('decision_sec')}s）"
                   if stats.get("decision_sec") is not None else ""))
            st.divider()
        try:
            from ashare_quant.fetchers import list_sources
            st.write("可用数据源：" + "、".join(list_sources()))
        except Exception:  # noqa: BLE001
            pass
        st.divider()
        parquet = list(data_dir.glob("*.parquet"))
        manifest2 = load_json(data_dir / "manifest.json")
        st.write(f"股票/指数缓存文件数：{len(parquet)}")
        st.write(f"数据清单（manifest）条目数：{len(manifest2) if manifest2 else 0}")
        if manifest2:
            stocks2 = {k: v for k, v in manifest2.items() if k != "sh000300"}
            idx_end2 = manifest2.get("sh000300", {}).get("end")
            stale2 = [k for k, v in stocks2.items()
                      if v.get("end") and idx_end2 and v["end"] < idx_end2]
            st.write(f"覆盖股票数：{len(stocks2)}")
            st.write(f"数据完整率：{(len(stocks2) - len(stale2)) / len(stocks2):.2%}"
                     f"（与指数同步 {len(stocks2) - len(stale2)}/{len(stocks2)}）")
            ends = {}
            for k, v in manifest2.items():
                if k == "sh000300":
                    continue
                ends.setdefault(v.get("end"), 0)
                ends[v["end"]] += 1
            st.write("股票数据截止日分布：" + "，".join(
                f"{d}:{n}只" for d, n in sorted(ends.items())))
        index_path = data_dir / "sh000300.parquet"
        if index_path.exists():
            idx = pd.read_parquet(index_path)
            st.write(f"沪深300 指数数据截止：{idx.index.max().date()}　行数：{len(idx)}")
        failed_path = data_dir / "update_failed.json"
        if failed_path.exists():
            failed = load_json(failed_path)
            if failed:
                st.warning(f"今日跳过（停牌/异常，次日自动重试）：{len(failed)} 只 "
                           f"— {'、'.join(list(failed)[:10])}")
    else:
        st.info("数据目录不存在，在「总览」点击「下载/更新全市场数据」。")

st.divider()
st.caption(DISCLAIMER)
