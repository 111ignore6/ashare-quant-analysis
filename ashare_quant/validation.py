from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationIssue:
    symbol: str
    rule: str
    detail: str


def validate_symbol(symbol: str, df: pd.DataFrame | None, calendar, missing_threshold: float = 0.05) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if df is None or df.empty:
        return [ValidationIssue(symbol, "empty", "no data")]
    if df.index.duplicated().any():
        issues.append(ValidationIssue(symbol, "duplicate_dates", str(int(df.index.duplicated().sum()))))
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        issues.append(ValidationIssue(symbol, "nonpositive_price", "price <= 0"))
    ret = df["close"].pct_change().dropna()
    if (ret.abs() > 0.21).any():
        issues.append(ValidationIssue(symbol, "limit_move", f"max |ret| {ret.abs().max():.3f}"))
    if calendar is not None:
        present = set(df.index)
        missing = [d for d in calendar.all_dates if d not in present]
        if missing and len(missing) / calendar.count() > missing_threshold:
            issues.append(ValidationIssue(symbol, "missing_dates", f"{len(missing)} missing"))
    return issues


def validate_panel(store, calendar, missing_threshold: float = 0.05) -> dict:
    report: dict[str, list[str]] = {}
    for symbol in store.symbols():
        issues = validate_symbol(symbol, store.load(symbol), calendar, missing_threshold)
        if issues:
            report[symbol] = [i.rule for i in issues]
    return report
