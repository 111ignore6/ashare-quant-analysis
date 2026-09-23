"""对比报告生成物的"不许说谎"测试。

背景（2026-09-18 审查发现）：`write_report` 曾把
「训练 **18** 个月」与「**沪深300 成分股** 3 年日线」两句话**写死**在字符串里，
而 `run_benchmark` 实际用 `train_months=15`，且全市场模式也照印"沪深300 成分股"
—— 生成物里的参数说明与真实运行不符。这类"文档说谎"没有任何测试会拦。
本文件把报告抬头钉在实际参数上。
"""

import numpy as np
import pandas as pd
from ashare_quant.ml.benchmark import BENCH_FOLDS, write_report


def _table() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "m1", "annual_return": 0.10, "sharpe": 1.0,
         "max_drawdown": -0.05, "win_rate": 0.5, "mean_ic": 0.01, "n_periods": 12},
    ])


def test_report_states_actual_fold_params_not_hardcoded(tmp_path):
    """报告必须写**实际跑**的折参数（默认 BENCH_FOLDS），不能写死 18 个月。"""
    out = tmp_path / "r.md"
    write_report(_table(), out, tmp_path / "r.json", **BENCH_FOLDS)
    text = out.read_text(encoding="utf-8")
    train = BENCH_FOLDS["train_months"]
    assert f"训练 {train} 个月" in text, f"报告必须写实际训练窗口（{train} 个月）"
    # 旧行为写死 18；只要它与实际值不同，就不允许出现在"训练 X 个月"里
    if train != 18:
        assert "训练 18 个月" not in text, "报告里的折参数写死了，与实际运行不符"


def test_report_states_actual_universe(tmp_path):
    """全市场模式不许照印"沪深300 成分股"。"""
    out = tmp_path / "r.md"
    write_report(_table(), out, tmp_path / "r.json", universe="等权全市场", **BENCH_FOLDS)
    text = out.read_text(encoding="utf-8")
    assert "等权全市场" in text
    assert "沪深300 成分股" not in text, "报告抬头的数据范围与实际运行不符"


def test_report_declares_known_biases(tmp_path):
    """报告必须自带已知偏差声明（净口径、beta、无 embargo）。"""
    out = tmp_path / "r.md"
    write_report(_table(), out, tmp_path / "r.json", **BENCH_FOLDS)
    text = out.read_text(encoding="utf-8")
    assert "净收益" in text
    assert "beta" in text, "必须声明这是绝对收益而非超额收益"
    assert "embargo" in text, "必须声明评估路径没有 purge/embargo"
    assert "n_periods" in text, "必须提示各行期数不同、夏普不可严格横比"


def test_report_does_not_hardcode_a_conclusion(tmp_path):
    """结论段不许是写死的断言，且必须带期数。"""
    out = tmp_path / "r.md"
    write_report(_table(), out, tmp_path / "r.json", **BENCH_FOLDS)
    text = out.read_text(encoding="utf-8")
    assert "期数 12" in text, "结论里必须带 n_periods，否则跨行比较会误导"
    assert "绝对收益（含 beta）" in text


def test_bench_folds_are_the_single_source_of_truth():
    """run_benchmark 与报告抬头必须共用 BENCH_FOLDS，不许各写一份。"""
    import inspect

    from ashare_quant.ml import benchmark

    src = inspect.getsource(benchmark.run_benchmark)
    assert "BENCH_FOLDS" in src, (
        "run_benchmark 必须用 BENCH_FOLDS 切折，否则折参数又会出现第二个副本")


def test_costs_are_threaded_into_ml_rows():
    """ML 行必须与基准行同口径（都扣成本），否则同表混排两种口径。"""
    import inspect

    from ashare_quant.ml import benchmark

    src = inspect.getsource(benchmark.run_benchmark)
    assert "costs=BENCH_COSTS" in src
    # ML 评估调用点必须显式传成本
    ml_call = src[src.index("walk_forward_ml_evaluate"):]
    assert "costs=BENCH_COSTS" in ml_call[:400], (
        "walk_forward_ml_evaluate 的调用点必须传 costs，否则 ML 行仍是毛收益")


def test_metrics_row_reports_period_count():
    """期数必须进表 —— 保形门控行只覆盖半个验证窗口，不比期数就会误导。"""
    from ashare_quant.ml.benchmark import _metrics_row

    row = _metrics_row("x", pd.Series(np.zeros(7)))
    assert row["n_periods"] == 7
