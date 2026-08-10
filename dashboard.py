"""Streamlit 仪表盘。

运行： python -m streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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


st.set_page_config(page_title="A股量化研究·模拟分析", layout="wide")
st.title("A股量化研究 · 模拟分析仪表盘")
st.caption(DISCLAIMER)

with st.sidebar:
    data_root = st.selectbox("数据范围", ["data/all", "data/3y"], index=0)
    mode = "all" if data_root == "data/all" else "3y"
    sim_dir = PROJECT / ("docs/simulation-all" if mode == "all" else "docs/simulation")
    st.caption(f"数据目录：{data_root}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["模拟盘", "今日决策", "算法对比", "调整日志", "数据状态"])

with tab1:
    st.subheader("模拟盘对比（月度调仓 Top-50，含交易成本）")
    st.caption("虚线为真实市场基准：等权全市场与沪深300 指数买入持有。")
    returns = load_csv(sim_dir / "model_returns.csv")
    if returns is None:
        st.info(f"未找到模拟盘结果，请先运行：`python -m ashare_quant.cli simulate --data-root {data_root}`")
    else:
        st.plotly_chart(equity_figure(returns), width='stretch')
        sim_json = load_json(sim_dir / "simulation.json")
        if sim_json and "summary" in sim_json:
            st.dataframe(format_metric(pd.DataFrame(sim_json["summary"])), width='stretch')

with tab2:
    st.subheader("今日模拟投资决策")
    decision = load_json(PROJECT / "docs/decision" / "decision.json")
    if decision is None:
        st.info(f"未找到决策结果，请先运行：`python -m ashare_quant.cli decision --data-root {data_root}`")
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
        st.dataframe(picks, width='stretch')
        st.caption("预期收益为多模型预测的未来 20 个交易日收益均值，模拟研究仅供学习。")

with tab3:
    st.subheader("算法表现对比（样本外夏普）")
    bench_stem = "algorithm-benchmark-all" if mode == "all" else "algorithm-benchmark"
    bench = load_json(PROJECT / "docs/research" / f"{bench_stem}.json")
    if bench is None:
        st.info("未找到算法对比结果，请先运行：`python -m ashare_quant.cli benchmark --data-root {0}`".format(data_root))
    else:
        df = pd.DataFrame(bench).sort_values("sharpe", ascending=False)
        fig = go.Figure(go.Bar(x=df["model"], y=df["sharpe"],
                               marker_color=["#c0392b" if v >= 2 else "#2980b9" for v in df["sharpe"]]))
        fig.update_layout(title="各算法样本外夏普", xaxis_title="模型", yaxis_title="夏普",
                          xaxis_tickangle=-30)
        fig.data[0].x = [display_name(m) for m in df["model"]]
        st.plotly_chart(fig, width='stretch')
        st.dataframe(format_metric(df), width='stretch')
        returns = load_csv(PROJECT / "docs/research" / f"{bench_stem}.returns.csv")
        if returns is not None:
            st.subheader("各算法样本外净值曲线（含真实基准）")
            st.caption("虚线为真实市场基准；曲线为月度调仓 Top-50 等权的样本外净值。")
            st.plotly_chart(equity_figure(returns), width='stretch')
        else:
            st.info("暂无收益曲线数据，重跑 `benchmark` 命令后自动生成。")

with tab4:
    st.subheader("反馈调整日志")
    LOG_NAMES = {"date": "日期", "trigger": "触发", "action": "动作",
                 "before": "调整前权重", "after": "调整后权重", "effect": "效果"}
    log_path = sim_dir / "adjustments.jsonl"
    if not log_path.exists():
        st.info("暂无调整日志。")
    else:
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        st.dataframe(pd.DataFrame(entries).rename(columns=LOG_NAMES), width='stretch')

with tab5:
    st.subheader("数据状态")
    data_dir = PROJECT / data_root
    if data_dir.exists():
        parquet = list(data_dir.glob("*.parquet"))
        manifest = load_json(data_dir / "manifest.json")
        st.write(f"股票/指数缓存文件数：{len(parquet)}")
        st.write(f"数据清单（manifest）条目数：{len(manifest) if manifest else 0}")
        if manifest:
            stocks = {k for k in manifest if k != "sh000300"}
            st.write(f"覆盖股票数：{len(stocks)}")
        index_path = data_dir / "sh000300.parquet"
        if index_path.exists():
            idx = pd.read_parquet(index_path)
            st.write(f"沪深300 指数数据截止：{idx.index.max().date()}　行数：{len(idx)}")
    else:
        st.info(f"数据目录不存在，请先运行：`python -m ashare_quant.cli fetch --data-root {data_root}`")

st.divider()
st.caption(DISCLAIMER)
