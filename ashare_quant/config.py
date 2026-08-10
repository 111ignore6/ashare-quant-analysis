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

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in names}
        if "data_root" in kwargs:
            kwargs["data_root"] = Path(kwargs["data_root"])
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        path = Path(path)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return cls.from_dict(yaml.safe_load(f) or {})
        return cls()
