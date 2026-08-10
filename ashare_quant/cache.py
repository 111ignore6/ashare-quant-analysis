from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def is_symbol_stem(stem: str) -> bool:
    """是否为股票/指数缓存文件（排除 features/panels 等缓存）。"""
    if len(stem) == 6 and stem.isdigit():
        return True
    return len(stem) > 2 and stem[:2] in ("sh", "sz", "bj") and stem[2:].isdigit()


class ParquetStore:
    """按 symbol 存 Parquet，manifest.json 记录区间与行数。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._manifest_lock = threading.Lock()

    def _path(self, symbol: str) -> Path:
        return self.root / f"{symbol}.parquet"

    def save(self, symbol: str, df: pd.DataFrame, update_manifest: bool = True) -> None:
        out = df.copy()
        out.index.name = "date"
        tmp = self._path(symbol).with_suffix(".parquet.tmp")
        out.to_parquet(tmp)
        os.replace(tmp, self._path(symbol))
        if update_manifest:
            self.update_manifest(symbol, out)

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self._path(symbol)
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def append(self, symbol: str, df: pd.DataFrame, update_manifest: bool = True) -> None:
        old = self.load(symbol)
        merged = df if old is None else pd.concat([old, df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.save(symbol, merged, update_manifest=update_manifest)

    def exists(self, symbol: str) -> bool:
        return self._path(symbol).exists()

    def symbols(self) -> list[str]:
        """数据目录里的标的代码（排除特征/面板等缓存文件）。"""
        out = []
        for p in self.root.glob("*.parquet"):
            stem = p.stem
            if is_symbol_stem(stem):
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

    def rebuild_manifest(self) -> dict:
        """遍历缓存目录一次性重建 manifest（批量下载后调用，替代逐条写入）。"""
        manifest = {}
        for p in self.root.glob("*.parquet"):
            if not is_symbol_stem(p.stem):
                continue  # 跳过 features/panels 等非行情缓存
            df = pd.read_parquet(p, columns=[])  # 仅取索引，不加载数据列
            manifest[p.stem] = {
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "rows": int(len(df)),
            }
        with self._manifest_lock:
            self.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
