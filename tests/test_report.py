import pandas as pd
from ashare_quant.research.report import build_report, conclusions


def _results():
    return {
        "distribution": {"kurtosis": 5.2, "skewness": -0.3, "n_symbols": 300},
        "volatility": {"ljungbox_p": 0.001, "arch_p": 0.002},
        "momentum": pd.DataFrame({"horizon": [5, 20, 60], "mean_ic": [-0.01, 0.02, 0.08]}),
        "factor_summary": pd.DataFrame({"icir": [0.5, 0.2]}, index=["momentum", "volume_ratio"]),
        "pca": {"first_ratio": 0.7, "n_components_80": 2},
        "regimes": pd.DataFrame({"state": [1, 2, 3], "mean_fwd": [0.01, 0.02, -0.03]}),
        "data_through": "2026-08-07",
    }


def test_conclusions_numbered():
    conc = conclusions(_results())
    ids = {c.split(" ")[0] for c in conc}
    assert ids == {"R1", "R2", "R3", "R4", "R5", "R6"}


def test_build_report_writes_file(tmp_path):
    out = tmp_path / "research.md"
    build_report(_results(), out)
    text = out.read_text(encoding="utf-8")
    assert "R1" in text and "不构成投资建议" in text and "2026-08-07" in text


def test_screening_markdown(tmp_path):
    from ashare_quant.research.report import screening_to_markdown

    df = pd.DataFrame({"model": ["benchmark", "reversal"], "params": ["-", "{'horizon': 60}"],
                       "sharpe": [0.5, 0.8], "max_drawdown": [-0.2, -0.1],
                       "keep": [True, True], "reason": ["基准", "样本外胜出"]})
    out = tmp_path / "model-selection.md"
    screening_to_markdown(df, out)
    text = out.read_text(encoding="utf-8")
    assert "reversal" in text and "不构成投资建议" in text
