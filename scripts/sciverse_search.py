"""SciVerse MCP 论文检索工具（stdio 直连，复用 ~/.codex/config.toml 中的 token）。

用法：
    python scripts/sciverse_search.py "stock prediction" --year-from 2023 --limit 5
    python scripts/sciverse_search.py "factor mining" --semantic
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time


class SciVerseClient:
    def __init__(self) -> None:
        cfg_path = os.path.expanduser("~/.codex/config.toml")
        cfg = open(cfg_path, encoding="utf-8").read()
        match = re.search(r'SCIVERSE_API_TOKEN\s*=\s*"([^"]+)"', cfg)
        if not match:
            raise RuntimeError("config.toml 中未找到 SCIVERSE_API_TOKEN")
        env = os.environ.copy()
        env["SCIVERSE_API_TOKEN"] = match.group(1)
        cmd = [r"D:\nodejs\node.exe", r"D:\nodejs\node_modules\npm\bin\npx-cli.js",
               "-y", "sciverse-mcp-server"]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", errors="replace", text=True, env=env, bufsize=1)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._id = 0
        self._init()

    def _drain_stderr(self) -> None:
        for _ in self.proc.stderr:
            pass

    def _init(self) -> None:
        self._call("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "codex-research", "version": "0.1"},
        })
        self._notify("notifications/initialized", {})

    def _send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _read_until_id(self, want: int, timeout: float = 90.0) -> dict | None:
        deadline = time.time() + timeout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if msg.get("id") == want:
                return msg
            if time.time() > deadline:
                return None
        return None

    def _call(self, method: str, params: dict) -> dict:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        resp = self._read_until_id(self._id)
        if resp is None:
            raise RuntimeError(f"调用 {method} 超时")
        if "error" in resp:
            raise RuntimeError(f"{method} 错误: {resp['error']}")
        return resp.get("result", {})

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _call_tool(self, name: str, arguments: dict) -> dict:
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content", [])
        text = "".join(item.get("text", "") for item in content if item.get("type") == "text")
        if text:
            try:
                return json.loads(text)
            except Exception:
                return {"raw_text": text}
        return result

    def search_papers(self, query: str, year_from: int | None = None,
                      limit: int = 10) -> list[dict]:
        params: dict = {"query": query, "page_size": limit}
        if year_from:
            params["year_from"] = year_from
        result = self._call_tool("search_papers", params)
        return result.get("papers", result.get("results", [])) or []

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        result = self._call_tool("semantic_search", {"query": query, "top_k": top_k})
        return result.get("chunks", result.get("results", [])) or []

    def close(self) -> None:
        self.proc.terminate()


def _year(paper: dict) -> str:
    for key in ("year", "pub_year", "publication_year", "publication_published_year",
                "date", "pub_date", "created"):
        value = paper.get(key)
        if value:
            return str(value)[:4]
    return "?"


def _author_names(authors) -> str:
    if not isinstance(authors, list):
        return str(authors) if authors else ""
    names = []
    for item in authors:
        if isinstance(item, dict):
            name = item.get("name") or item.get("author") or ""
        else:
            name = str(item)
        if name:
            names.append(name)
    return ", ".join(names[:4])


def _paper_line(paper: dict) -> str:
    title = paper.get("title") or paper.get("name") or "?"
    authors = _author_names(paper.get("authors") or paper.get("author"))
    ident = paper.get("url") or paper.get("doi") or paper.get("id") or paper.get("unique_id") or ""
    return f"- [{_year(paper)}] {title} | {authors} | {ident}"


def main() -> None:
    parser = argparse.ArgumentParser(description="SciVerse 论文检索")
    parser.add_argument("query")
    parser.add_argument("--year-from", type=int)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()

    client = SciVerseClient()
    try:
        if args.semantic:
            items = client.semantic_search(args.query, top_k=args.limit)
            for item in items:
                text = item.get("content") or item.get("text") or str(item)
                print("-", text[:300].replace("\n", " "))
        else:
            papers = client.search_papers(args.query, year_from=args.year_from, limit=args.limit)
            for paper in papers:
                print(_paper_line(paper))
            print(f"\n共 {len(papers)} 条")
    finally:
        client.close()


if __name__ == "__main__":
    main()
