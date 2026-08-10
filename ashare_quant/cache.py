from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


class ParquetStore:
    """按 symbol 存 Parquet，manifest.json 记录区间与行数。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._manifest_lock = threading.Lock()

    def _path(self, symbol: str) -> Path:
        return self.root / f"{symbol}.parquet"

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        out = df.copy()
        out.index.name = "date"
        tmp = self._path(symbol).with_suffix(".parquet.tmp")
        out.to_parquet(tmp)
        os.replace(tmp, self._path(symbol))
        self.update_manifest(symbol, out)

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self._path(symbol)
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def append(self, symbol: str, df: pd.DataFrame) -> None:
        old = self.load(symbol)
        merged = df if old is None else pd.concat([old, df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.save(symbol, merged)

    def exists(self, symbol: str) -> bool:
        return self._path(symbol).exists()

    def symbols(self) -> list[str]:
        """数据目录里的标的代码（排除特征/面板等缓存文件）。"""
        out = []
        for p in self.root.glob("*.parquet"):
            stem = p.stem
            if len(stem) == 6 and stem.isdigit():
                out.append(stem)
            elif len(stem) > 2 and stem[:2] in ("sh", "sz", "bj") and stem[2:].isdigit():
                out.append(stem)
        return sorted(out)

    def read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def update_manifest(self, symbol: str, df: pd.DataFrame) -> None:
        with self._manifest_lock:
            m = self.read_manifest()
            m[symbol] = {
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "rows": int(len(df)),
            }
            self.manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
