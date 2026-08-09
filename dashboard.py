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
        fig.add_trace(go.Scatter(x=equity.index, y=equity[col], mode="lines", name=col))
    fig.update_layout(title="模型净值曲线（模拟）", xaxis_title="日期", yaxis_title="净值")
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
    returns = load_csv(sim_dir / "model_returns.csv")
    if returns is None:
        st.info("未找到模拟盘结果，请先运行：`python -m ashare_quant.cli simulate --data-root {0}`".format(data_root))
    else:
        st.plotly_chart(equity_figure(returns), use_container_width=True)
        sim_json = load_json(sim_dir / "simulation.json")
        if sim_json and "summary" in sim_json:
            st.dataframe(pd.DataFrame(sim_json["summary"]), use_container_width=True)

with tab2:
    st.subheader("今日模拟投资决策")
    decision = load_json(PROJECT / "docs/decision" / "decision.json")
    if decision is None:
        st.info("未找到决策结果，请先运行：`python -m ashare_quant.cli decision --data-root {0}`".format(data_root))
    else:
        st.write(f"决策日期：{decision['date']}　持仓 {len(decision['picks'])} 只　"
                 f"模型：{'、'.join(decision['models'])}")
        st.dataframe(pd.DataFrame(decision["picks"]), use_container_width=True)

with tab3:
    st.subheader("算法表现对比（样本外夏普）")
    bench = load_json(PROJECT / ("docs/research/algorithm-benchmark-all.json"
                                 if mode == "all" else "docs/research/algorithm-benchmark.json"))
    if bench is None:
        st.info("未找到算法对比结果，请先运行 `benchmark` 命令。")
    else:
        df = pd.DataFrame(bench).sort_values("sharpe", ascending=False)
        fig = go.Figure(go.Bar(x=df["model"], y=df["sharpe"],
                               marker_color=["#c0392b" if v >= 2 else "#2980b9" for v in df["sharpe"]]))
        fig.update_layout(title="各算法样本外夏普", xaxis_title="模型", yaxis_title="夏普")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

with tab4:
    st.subheader("反馈调整日志")
    log_path = sim_dir / "adjustments.jsonl"
    if not log_path.exists():
        st.info("暂无调整日志。")
    else:
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        st.dataframe(pd.DataFrame(entries), use_container_width=True)

with tab5:
    st.subheader("数据状态")
    data_dir = PROJECT / data_root
    if data_dir.exists():
        parquet = list(data_dir.glob("*.parquet"))
        manifest = load_json(data_dir / "manifest.json")
        st.write(f"股票/指数 parquet 文件数：{len(parquet)}")
        st.write(f"manifest 条目数：{len(manifest) if manifest else 0}")
        index_path = data_dir / "sh000300.parquet"
        if index_path.exists():
            idx = pd.read_parquet(index_path)
            st.write(f"沪深300指数数据截止：{idx.index.max().date()}")
    else:
        st.info("数据目录不存在，请先运行 fetch。")

st.divider()
st.caption(DISCLAIMER)
