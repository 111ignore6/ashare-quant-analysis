from __future__ import annotations

import pandas as pd

from ..research.factors import compute_factors, winsorize_zscore
from .base import Model


class ReversalModel(Model):
    """均值回归：过去 h 日跌得越多的股票得分越高。"""
    name = "reversal"

    def __init__(self, horizon: int = 60) -> None:
        self.horizon = horizon

    def score(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        return -close.pct_change(self.horizon, fill_method=None)


class LowVolModel(Model):
    """低波动防守：滚动波动率越低得分越高。"""
    name = "lowvol"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def score(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        vol = close.pct_change(fill_method=None).rolling(self.window).std()
        return -vol


class MomentumModel(Model):
    """动量趋势（对照组）：过去 h 日涨幅越高得分越高。"""
    name = "momentum"

    def __init__(self, horizon: int = 20) -> None:
        self.horizon = horizon

    def score(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        return close.pct_change(self.horizon, fill_method=None)


class MultiFactorModel(Model):
    """横截面多因子：有效因子 z-score 加权求和。"""
    name = "multifactor"

    FACTORS = ("volume_ratio", "ma_deviation", "reversal60", "lowvol")

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {f: 1 / len(self.FACTORS) for f in self.FACTORS}

    def score(self, close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        base = compute_factors(close, volume)
        custom = {
            "volume_ratio": base["volume_ratio"],
            "ma_deviation": base["ma_deviation"],
            "reversal60": -close.pct_change(60, fill_method=None),
            "lowvol": -close.pct_change(fill_method=None).rolling(20).std(),
        }
        total = None
        for name, w in self.weights.items():
            z = winsorize_zscore(custom[name])
            total = w * z if total is None else total + w * z
        return total
