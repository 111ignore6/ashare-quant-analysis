from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    data_root: Path = Path("data")
    universe_mode: str = "csi300"
    data_source: str = "tencent"
    fallback_source: str = "akshare"
    fallback_sources: tuple[str, ...] = ("tencent", "akshare")
    years: int = 3
    adjust: str = "qfq"
    # 每日决策模型池；空列表 = 默认 6 模型（lgbm/histgb/rf/svm/knn/linear）
    models: tuple[str, ...] = ()
    max_workers: int = 8
    retry: int = 3
    rate_limit_per_second: float = 2.0
    top_n: int = 50
    rebalance: str = "M"
    # 交易成本：账户净值在每次换仓时按这三项扣费（默认值与 backtest/engine.py 一致）。
    # 2026-09-16 之前账户一分钱成本都没扣，而换手是每日级（单边约 59%），
    # 实测成本一项就会吃掉约 76% 的账面收益 —— 不扣等于系统性高估。
    commission: float = 0.00025   # 佣金（双边）
    stamp: float = 0.0005         # 印花税（仅卖出）
    slippage: float = 0.001       # 滑点（双边）
    # 训练目标口径：raw=未来 20 日原始收益（历史口径）；excess=同一天横截面去均值。
    # 策略是横截面 Top-N 且永远满仓，市场共同波动那一块选谁都一样 —— 用 raw 训练
    # 会让模型把容量花在当天排名用不到的成分上。2026-09-16 A/B 实测（walk-forward）：
    # excess 在两个可比较的回归模型上都更好（lgbm −0.56%→−0.09%、huber −0.92%→+0.13%
    # 日均截面超额），rank_lgb 因只用当日相对顺序而完全不变（机制自证）。
    # **但默认仍为 raw**：有效独立样本仅约 12 个（20 日前瞻收益高度重叠），
    # 证据是"方向一致的建议"而非"已证实"，故不擅自改动生产模型。要切换改这一行为 excess。
    target_mode: str = "raw"
    initial_capital: float = 100000.0
    stop_loss: float | None = None
    take_profit: float | None = None
    auto_update: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        if "data_root" in kwargs:
            kwargs["data_root"] = Path(kwargs["data_root"])
        if "fallback_sources" in kwargs and isinstance(kwargs["fallback_sources"], (list, tuple)):
            kwargs["fallback_sources"] = tuple(str(x) for x in kwargs["fallback_sources"])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        """序列化为可写回 yaml 的字典（Path 转字符串）。"""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["data_root"] = str(out["data_root"])
        return out

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        path = Path(path)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return cls.from_dict(yaml.safe_load(f) or {})
        return cls()


def _coerce(field_name: str, value):
    """按 dataclass 字段类型做宽松强转（None 原样保留）。"""
    if value is None:
        return None
    types = {f.name: f.type for f in fields(Config)}
    t = types.get(field_name)
    if t is bool:
        return bool(value)
    if t is float:
        return float(value)
    if t is int:
        return int(value)
    if t is str:
        return str(value)
    if getattr(t, "__origin__", None) is tuple:
        if isinstance(value, (list, tuple)):
            return tuple(str(x) for x in value)
    return value


def update_config_yaml(path: Path, **kwargs) -> dict:
    """合并写回配置：只更新指定字段，其余字段保留原值。"""
    path = Path(path)
    merged: dict = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            merged = yaml.safe_load(f) or {}
    names = {f.name for f in fields(Config)}
    for k, v in kwargs.items():
        if k in names:
            merged[k] = _coerce(k, v)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(merged, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return merged
