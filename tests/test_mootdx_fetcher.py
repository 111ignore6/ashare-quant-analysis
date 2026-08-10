import numpy as np
import pandas as pd

from ashare_quant.fetchers.mootdx_fetcher import _qfq_adjust, _to_tdx_code


def test_to_tdx_code():
    assert _to_tdx_code("600000") == "600000"
    assert _to_tdx_code("000001") == "000001"


def test_qfq_adjust_dividend():
    """10 派 4.2 元（每股 0.42）：除权日后价格不变，之前按理论因子缩放。"""
    idx = pd.to_datetime(["2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17"])
    bars = pd.DataFrame({
        "open": [9.20, 9.08, 8.92, 8.85],
        "high": [9.22, 9.31, 8.95, 8.88],
        "low": [9.10, 9.00, 8.80, 8.82],
        "close": [9.16, 9.31, 8.85, 8.87],
        "volume": [1e6] * 4,
        "amount": [9e6] * 4,
    }, index=idx)
    xdxr = pd.DataFrame([{
        "year": 2026, "month": 7, "day": 16,
        "fenhong": 4.2, "songzhuangu": 0.0, "peigu": 0.0, "peigujia": 0.0,
    }])
    out = _qfq_adjust(bars, xdxr)
    # 除权日（7-16）及之后：因子 1
    assert out.loc["2026-07-16", "close"] == 8.85
    assert out.loc["2026-07-17", "close"] == 8.87
    # 除权日前（7-15 及之前）：因子 = (9.31 - 0.42)/9.31
    expected = (9.31 - 0.42) / 9.31
    assert abs(out.loc["2026-07-15", "close"] - 9.31 * expected) < 1e-9
    assert abs(out.loc["2026-07-14", "close"] - 9.16 * expected) < 1e-9
    # 成交量不受复权影响
    assert out.loc["2026-07-15", "volume"] == 1e6


def test_qfq_adjust_empty_xdxr():
    idx = pd.to_datetime(["2026-07-14", "2026-07-15"])
    bars = pd.DataFrame({
        "open": [9.2, 9.08], "high": [9.22, 9.31],
        "low": [9.1, 9.0], "close": [9.16, 9.31],
        "volume": [1e6, 1e6], "amount": [9e6, 9e6],
    }, index=idx)
    out = _qfq_adjust(bars, pd.DataFrame())
    assert np.allclose(out["close"], [9.16, 9.31])
