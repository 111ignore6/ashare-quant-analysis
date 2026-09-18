"""成交量完整性审计脚本的判据回归（2026-09-18）。

这里钉住的核心是一条**实测得出的事实**：腾讯的 ``amount`` 是
``volume × 调整后收盘`` 的**估算式**，不是成交额真值
（实测 600000/000001 各 1628 日里与 akshare 分别有 1101/1094 日不符，中位比 1.25）。
所以 ``scripts/audit_volume_integrity.py`` 只允许用腾讯的 **volume** 做独立确证，
真值一律取 akshare；否则会把好行改坏。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "audit_volume_integrity", ROOT / "scripts" / "audit_volume_integrity.py")
avi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(avi)

D0 = pd.Timestamp("2025-01-03")


def _frame(volume: float, amount: float, day: pd.Timestamp = D0) -> pd.DataFrame:
    return pd.DataFrame({"volume": [volume], "amount": [amount]}, index=[day])


def _diff(local: pd.DataFrame, ak: pd.DataFrame) -> pd.DataFrame:
    return avi.compare(local, ak)


def test_compare_ignores_agreement():
    local = _frame(1_000_000, 10_000_000)
    assert _diff(local, _frame(1_000_000, 10_000_000)).empty
    # ±12% 容差内的正常抖动不算异常（换手统计口径差异）
    assert _diff(local, _frame(1_050_000, 9_700_000)).empty


def test_compare_flags_volume_and_amount_separately():
    out = _diff(_frame(46_400, 59_059), _frame(5_900, 59_059))
    assert len(out) == 1 and bool(out.iloc[0]["bad_vol"]) and not bool(out.iloc[0]["bad_amt"])
    out = _diff(_frame(1_000_000, 12_000_000), _frame(1_000_000, 9_600_000))
    assert len(out) == 1 and not bool(out.iloc[0]["bad_vol"]) and bool(out.iloc[0]["bad_amt"])


def test_compare_treats_zero_volume_as_not_bad():
    """停牌行 volume=0 → 比值无意义，不得当成异常（否则每次审计都刷出满屏噪音）。"""
    out = _diff(_frame(0, 0), _frame(0, 0))
    assert out.empty


def test_confirmed_uses_akshare_values_not_tencent():
    """真实案例 603559@2025-01-03：本地 46,400，akshare/腾讯 volume 一致为 5,900。"""
    local, ak, tc = _frame(46_400, 59_059), _frame(5_900, 59_059), _frame(5_900, 59_059)
    rows = avi.adjudicate("603559", local, ak, tc, _diff(local, ak))
    assert len(rows) == 1
    e = rows[0]
    assert e["kind"] == "confirmed"
    assert e["fix"]["volume_true"] == 5_900        # 真值来自 akshare
    assert e["fix"]["amount_true"] == 59_059
    assert e["fix"]["fix_amount"] is False         # amount 本来就对 → 不动它


def test_amount_estimate_never_drives_a_rewrite():
    """本脚本存在的理由：只有成交额不符时，腾讯的估算 amount 绝不能当"真值"写回去。

    本地 12,000,000 / akshare 9,600,000（不符）/ 腾讯 12,000,000（= 本地，估算式）。
    旧判据会因为"本地与腾讯一致"而判 confirmed，并把腾讯的估算 amount 写进 parquet；
    新判据只报告不改。
    """
    local = _frame(1_000_000, 12_000_000)
    ak = _frame(1_000_000, 9_600_000)
    tc = _frame(1_000_000, 12_000_000)
    rows = avi.adjudicate("600000", local, ak, tc, _diff(local, ak))
    assert len(rows) == 1
    assert rows[0]["kind"] == "amount_only"
    assert "fix" not in rows[0]


def test_local_agreeing_with_tencent_only_is_not_confirmed():
    """本地与腾讯一致、但与 akshare 不符 = 一比一，无法仲裁 → 只报告。"""
    local, ak, tc = _frame(5_900, 59_059), _frame(46_400, 59_059), _frame(5_900, 59_059)
    rows = avi.adjudicate("603559", local, ak, tc, _diff(local, ak))
    assert [r["kind"] for r in rows] == ["sources_disagree"]


def test_missing_tencent_coverage_is_reported_only():
    """北交所等腾讯无覆盖的标的：没有第二确证 → single_source，不改数据。"""
    local, ak = _frame(46_400, 59_059), _frame(5_900, 59_059)
    for tc in (None, _frame(1, 1, pd.Timestamp("2030-01-01"))):   # 无数据 / 无重叠日
        rows = avi.adjudicate("920100", local, ak, tc, _diff(local, ak))
        assert [r["kind"] for r in rows] == ["single_source"]
        assert "fix" not in rows[0]


def test_apply_fixes_and_idempotency(tmp_path):
    """写入只看 akshare 值；改完复测必须回到"零异常"（幂等）。"""
    sym = "603559"
    p = tmp_path / f"{sym}.parquet"
    good = _frame(5_900, 59_059, D0)
    bad_row = pd.DataFrame({"volume": [46_400], "amount": [12_000_000]}, index=[D0])
    pd.concat([bad_row, good.set_index(pd.Index([pd.Timestamp("2025-01-06")]))]).to_parquet(p)

    local = pd.read_parquet(p)
    ak = pd.DataFrame({"volume": [5_900, 5_900], "amount": [59_059, 59_059]},
                      index=[D0, pd.Timestamp("2025-01-06")])
    tc = pd.DataFrame({"volume": [5_900, 5_900], "amount": [999_999, 999_999]},
                      index=[D0, pd.Timestamp("2025-01-06")])
    rows = avi.adjudicate(sym, local, ak, tc, _diff(local, ak))
    assert [r["kind"] for r in rows] == ["confirmed"]

    bak = tmp_path / "bak"
    assert avi.apply_fixes(tmp_path, rows, bak) == 1
    assert (bak / f"{sym}.parquet").exists()          # 写入前必须留备份
    fixed = pd.read_parquet(p)
    assert fixed.loc[D0, "volume"] == 5_900
    # amount 那一行本地是错值、akshare 是 59,059 → 按 fix_amount 一并改
    assert fixed.loc[D0, "amount"] == 59_059
    assert fixed.loc[pd.Timestamp("2025-01-06"), "amount"] == 59_059  # 好行不动

    assert avi.compare(fixed, ak).empty               # 幂等：复测零异常
