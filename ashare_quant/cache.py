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
        # 内容没变就**不写盘**（2026-09-18 修）：`daily` 每次都会
        # `store.append(index_symbol, idx_df)`，即使指数没有新交易日，旧写法也会重写
        # 指数 parquet → mtime/size 变化 → `pipeline._source_signature` 随之变化
        # → 面板缓存**每次运行都判失效**（重建 ~25 秒）、特征表也跟着重建
        # （实测 17:20 与 18:01 两次运行都打印"本次重建特征表"，而面板内容完全一致）。
        # 只比"内容"不比"是否调用过"：真正的数值/行数变化仍会写盘 → 签名照样变化，
        # 判据强度不变（这正是 source_signature 存在的理由）。
        if old is not None and old.equals(merged):
            # 内容没变 → 不碰 parquet（保住 mtime，缓存判据才不会被自己刷失效）。
            # 但 manifest 仍要保证是最新的：手工删过 manifest.json 时，
            # 这里若不补写，指数就会永远缺 manifest 条目（下游按 manifest 判日期）。
            if update_manifest:
                self.update_manifest(symbol, merged)
            return
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
