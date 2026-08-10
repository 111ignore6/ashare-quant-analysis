"""A股量化研究·模拟分析控制台（Streamlit 单页应用）。

运行： python -m streamlit run dashboard.py
功能：状态总览 / 数据���载与更新（后台任务+实时输出）/ 模拟盘 / 今日决策 / 算法对比 / 日志。
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh

from ashare_quant.account import account_snapshot
from ashare_quant.realtime import snapshot

PROJECT = Path(__file__).parent
DISCLAIMER = "模拟研究，仅用于数据分析与学习，不构成投资建议。"

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


@st.cache_data(ttl=600)
def load_panel_close(data_dir: Path):
    """读取面板缓存中的收盘价矩阵（date × symbol）。"""
    p = data_dir / "panels" / "close.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


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

_TASK_QUEUES: dict[str, queue.Queue] = {}


def _task_worker(key: str, cmd: list[str], cwd: Path) -> None:
    q = _TASK_QUEUES[key]
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            q.put(("line", line.rstrip()))
        proc.wait()
        q.put(("done", proc.returncode))
    except Exception as e:  # noqa: BLE001
        q.put(("line", f"启动失败：{e}"))
        q.put(("done", -1))


def launch_task(key: str, cmd: list[str], cwd: Path) -> None:
    """启动一个后台任务（幂等：同一 key 运行中不重复启动）。"""
    state = st.session_state.setdefault("task_state", {}).setdefault(
        key, {"lines": [], "running": False, "code": None})
    if state["running"]:
        return
    _TASK_QUEUES[key] = queue.Queue()
    state.update({"lines": [], "running": True, "code": None})
    threading.Thread(target=_task_worker, args=(key, cmd, cwd), daemon=True).start()


def render_task(key: str, title: str) -> bool:
    """渲染任务进度；返回是否仍在运行。"""
    state = st.session_state.setdefault("task_state", {}).setdefault(
        key, {"lines": [], "running": False, "code": None})
    q = _TASK_QUEUES.get(key)
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
        with st.status(f"{title} 进行中…", expanded=True) as status:
            tail = state["lines"][-30:]
            st.code("\n".join(tail) if tail else "等待输出…（长任务请耐心等待）")
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
    decision = load_json(PROJECT / "docs/decision" / "decision.json")
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
    account = account_snapshot(decision, close_panel) \
        if decision and close_panel is not None else None
    if account:
        st.subheader("模拟账户")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("初始资金", f"{account['initial']:,.0f} 元")
        a2.metric("总资产", f"{account['total_asset']:,.0f} 元")
        a3.metric("总收益率", f"{account['total_return']:+.2%}")
        a4.metric("浮动盈亏", f"{account['total_pnl']:+,.0f} 元")
        st.caption(f"按决策日 {decision['date']} 收盘买入、最新收盘价 {account['as_of']} 估值；"
                   "模拟研究，不构成投资建议。")

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
        cols = st.columns(len(cum))
        for col, (name, v) in zip(cols, cum.items()):
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
            st.caption("每只股票在 LGBM / 梯度提升 / SVM 三个模型下的未来 20 日预期收益，"
                       "最终得分为三模型均值经置信度加权。")
            detail = pd.DataFrame(decision["picks"]).copy()
            if "model_scores" in detail.columns and detail["model_scores"].notna().any():
                scores = pd.json_normalize(detail["model_scores"].dropna().tolist())
                detail = pd.concat([detail[["symbol", "score"]], scores], axis=1)
                detail = detail.rename(columns={"symbol": "代码", "score": "加权得分"})
                st.dataframe(detail, width="stretch")
            else:
                st.info("当前决策文件缺少模型明细，重新运行 daily 后自动生成。")

with tab_account:
    st.subheader("模拟账户净值（自首个正式决策日跟踪）")
    st.caption("账户自首个正式决策日（样本外）开始逐日盯市值；"
               "决策每日收盘后生成，收益随每日更新持续累积。"
               "历史策略表现请参考「模拟盘」页。")
    equity = load_csv(data_dir / "portfolio" / "account_equity.csv")
    acc_summary = load_json(data_dir / "portfolio" / "account_summary.json")
    if equity is None or equity.empty:
        st.info("暂无账户曲线，运行「每日更新」生成首个正式决策后开始记录。")
    else:
        col = equity.columns[0]
        initial = float(acc_summary["initial_capital"]) if acc_summary else 100000.0
        start = equity.index[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity[col], mode="lines",
                                 name="账户净值", line=dict(color="#2980b9")))
        # 真实基准：沪深300 与等权全市场（同起点归一化到初始资金）
        idx_path = data_dir / "sh000300.parquet"
        if idx_path.exists():
            idx_close = pd.read_parquet(idx_path)["close"]
            idx_sel = idx_close.loc[start:]
            if len(idx_sel) >= 2:
                bench = initial * idx_sel / idx_sel.iloc[0]
                fig.add_trace(go.Scatter(x=bench.index, y=bench, name="基准·沪深300",
                                         line=dict(dash="dash", color="#7f8c8d")))
        if close_panel is not None and start in close_panel.index:
            eq_ret = close_panel.loc[start:].mean(axis=1)
            if len(eq_ret) >= 2:
                bench2 = initial * eq_ret / eq_ret.iloc[0]
                fig.add_trace(go.Scatter(x=bench2.index, y=bench2,
                                         name="基准·等权全市场",
                                         line=dict(dash="dash", color="#95a5a6")))
        fig.update_layout(title="账户净值曲线（元，虚线为真实基准）", xaxis_title="日期",
                          yaxis_title="总资产（元）", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        if len(equity) < 5:
            st.info("账户刚刚开始记录（当前仅 1 个交易日），曲线会随每日更新逐步成形；"
                    "想看完整历史策略表现，请切换到「模拟盘」页。")
        if acc_summary:
            a1, a2, a3, a4, a5 = st.columns(5)
            a1.metric("总资产", f"{acc_summary['total_asset']:,.0f} 元")
            a2.metric("总收益率", f"{acc_summary['total_return']:+.2%}")
            a3.metric("年化收益", f"{acc_summary['annual_return']:+.2%}")
            a4.metric("夏普", f"{acc_summary['sharpe']:.2f}")
            a5.metric("最大回撤", f"{acc_summary['max_drawdown']:.2%}")
            st.caption(f"已记录 {acc_summary['decisions']} 次决策，"
                       f"数据截至 {acc_summary['as_of']}。")
        csv_data = equity.to_csv().encode("utf-8-sig")
        st.download_button("下载账户净值 CSV", data=csv_data,
                           file_name="account_equity.csv", mime="text/csv")

with tab_realtime:
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
            snap = snapshot(symbols)
            if snap.empty:
                st.warning("未获取到实时行情（可能非交易时段或接口限流），"
                           "可切换数据源重试。")
            else:
                close_panel = load_panel_close(data_dir)
                if close_panel is not None and pd.Timestamp(decision["date"]) in close_panel.index:
                    base = close_panel.loc[pd.Timestamp(decision["date"]), snap["代码"]]
                    snap["决策日收盘"] = base.to_numpy()
                    snap["自决策日涨跌"] = snap["现价"] / snap["决策日收盘"] - 1
                    snap["自决策日涨跌"] = snap["自决策日涨跌"].map(
                        lambda v: f"{v:+.2%}" if pd.notna(v) else "-")
                    snap["决策日收盘"] = snap["决策日收盘"].map(
                        lambda v: f"{v:.2f}" if pd.notna(v) else "-")
                snap["涨跌幅"] = snap["涨跌幅"].map(lambda v: f"{v:+.2%}")
                for col in ("现价", "今开", "最高", "最低", "昨收"):
                    if col in snap.columns:
                        snap[col] = snap[col].map(
                            lambda v: f"{v:.2f}" if pd.notna(v) else "-")
                st.dataframe(snap, width="stretch")
                st.caption(f"决策日期 {decision['date']}，共 {len(snap)} 只；"
                           "「自决策日涨跌」为现价相对决策日收盘的变化。")
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
        parquet = list(data_dir.glob("*.parquet"))
        manifest2 = load_json(data_dir / "manifest.json")
        st.write(f"股票/指数缓存文件数：{len(parquet)}")
        st.write(f"数据清单（manifest）条目数：{len(manifest2) if manifest2 else 0}")
        if manifest2:
            st.write(f"覆盖股票数：{len({k for k in manifest2 if k != 'sh000300'})}")
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
