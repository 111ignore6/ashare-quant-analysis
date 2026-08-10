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
    years: int = 3
    adjust: str = "qfq"
    max_workers: int = 8
    retry: int = 3
    rate_limit_per_second: float = 2.0
    top_n: int = 50
    rebalance: str = "M"
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
