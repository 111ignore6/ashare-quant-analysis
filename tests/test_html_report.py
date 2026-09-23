import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ashare_quant.report.html_report import build_html_report, drawdown_figure, equity_figure, factor_heatmap


def _returns():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-31", periods=12, freq="ME")
    return pd.DataFrame({"momentum": rng.normal(0.01, 0.03, 12),
                         "rotation": rng.normal(0.008, 0.02, 12)}, index=idx)


def test_figures_are_plotly():
    r = _returns()
    assert isinstance(equity_figure(r), go.Figure)
    assert isinstance(drawdown_figure(r), go.Figure)
    assert isinstance(factor_heatmap(pd.DataFrame({"icir": [0.5, -0.3]}, index=["a", "b"])), go.Figure)


def test_build_html_report_writes_file(tmp_path):
    out = tmp_path / "report.html"
    build_html_report(equity_figure(_returns()), drawdown_figure(_returns()),
                      factor_heatmap(pd.DataFrame({"icir": [0.5]}, index=["a"])),
                      [{"date": "2026-08-07", "trigger": "rotation", "action": "weights",
                        "before": {}, "after": {}, "effect": "轮动"}],
                      data_through="2026-08-07", path=out)
    text = out.read_text(encoding="utf-8")
    assert "plotly" in text and "不构成投资建议" in text and "2026-08-07" in text
