from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
import pandas as pd

from .ops import winsorize_zscore_np


_FACTOR_REGISTRY: dict[str, Callable] = {}


def register_factor(name: str | None = None):
    """注册一个技术因子；签名建议 (close, volume, **params)，参数按需注入。

    usage:
        @register_factor("my_factor")
        def my_factor(close, volume, n_long=20):
            return ...
    """
    def deco(fn: Callable) -> Callable:
        _FACTOR_REGISTRY[name or fn.__name__] = fn
        return fn
    return deco


def _call_factor(fn: Callable, close: pd.DataFrame, volume: pd.DataFrame,
                 params: dict) -> pd.DataFrame:
    """只传入函数声明接受的参数，便于第三方因子自定义签名。"""
    sig = inspect.signature(fn)
    kwargs = {k: v for k, v in params.items() if k in sig.parameters}
    return fn(close, volume, **kwargs)


@register_factor("momentum")
def _momentum(close: pd.DataFrame, volume: pd.DataFrame, n_long: int = 20) -> pd.DataFrame:
    return close.pct_change(n_long, fill_method=None)


@register_factor("reversal")
def _reversal(close: pd.DataFrame, volume: pd.DataFrame, n_short: int = 5) -> pd.DataFrame:
    return -close.pct_change(n_short, fill_method=None)


@register_factor("volatility")
def _volatility(close: pd.DataFrame, volume: pd.DataFrame, n_vol: int = 20) -> pd.DataFrame:
    return close.pct_change(fill_method=None).rolling(n_vol).std()


@register_factor("ma_deviation")
def _ma_deviation(close: pd.DataFrame, volume: pd.DataFrame, n_long: int = 20) -> pd.DataFrame:
    return close / close.rolling(n_long).mean() - 1


@register_factor("volume_ratio")
def _volume_ratio(close: pd.DataFrame, volume: pd.DataFrame,
                  n_short: int = 5, n_long: int = 20) -> pd.DataFrame:
    return volume.rolling(n_short).mean() / volume.rolling(n_long).mean()


def compute_factors(close: pd.DataFrame, volume: pd.DataFrame,
                    n_short: int = 5, n_long: int = 20, n_vol: int = 20) -> dict[str, pd.DataFrame]:
    """按注册表顺序计算全部技术因子（未标准化）。新增因子只需 @register_factor。"""
    params = {"n_short": n_short, "n_long": n_long, "n_vol": n_vol}
    return {name: _call_factor(fn, close, volume, params)
            for name, fn in _FACTOR_REGISTRY.items()}


def list_factors() -> list[str]:
    """列出已注册的因子名。"""
    return list(_FACTOR_REGISTRY)


def winsorize_zscore(df: pd.DataFrame, clip: float = 0.01) -> pd.DataFrame:
    """逐日横截面：先去极值（分位截断），再转 z-score（总体标准差，保证每行均值0、标准差1）。"""
    return pd.DataFrame(
        winsorize_zscore_np(df.to_numpy(dtype=float), clip=clip),
        index=df.index, columns=df.columns)
