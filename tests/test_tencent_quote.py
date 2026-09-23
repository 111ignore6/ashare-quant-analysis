"""腾讯批量报价端点（`fetchers/tencent_quote.py`）的回归 —— 2026-09-18 新增。

**为什么这个文件以前不存在是个问题**：每日增量的快路径（一次请求 120 只、实测
310~340 只/s）就是走这个模块，而它此前**没有任何测试**；模块文档里"volume 按手给、
×100 得股"的结论只在新浪/主板样本上验证过 20 只，**样本里没有科创板**。
后果：2026-09-18 当日**全市场科创板**成交量被放大 100 倍写进 parquet
（688525 写成 1,889,345,500，真值 18,893,455），第二天才被成交量审计脚本发现。

下面的文本是 2026-09-18 16:5x 从 `https://qt.gtimg.cn/q=sh688525,sh600000` 实测抓下的原文。
"""

from __future__ import annotations

from ashare_quant.fetchers import tencent_quote as tq

# 实测原文（字段用 '~' 分隔；第 7 个字段是成交量、第 36 个字段是 "现价/量/额"）
REAL = (
    'v_sh688525="1~XD佰维存~688525~216.88~208.27~215.00~18893455~10323187~8570268~216.87~35~'
    '216.85~9~216.84~19~216.83~77~216.82~2~216.88~29~216.89~4~216.90~6~216.92~10~216.94~4~~'
    '20260918161448~8.61~4.13~218.60~211.23~216.88/18893455/4076690406~18893455~407669~3.96~'
    '12.54~~218.60~211.23~3.54~1034.15~1034.15~8.22~249.92~166.62~1.36~89~215.77~7.22~121.23~'
    '~~3.12~407669.0406~259.8873~11983~A RA~GP-A-KCB";\n'
    'v_sh600000="1~浦发银行~600000~9.07~9.06~9.05~517593~277236~240357~9.07~4467~9.06~11048~'
    '9.05~7788~9.04~5161~9.03~4586~9.08~2564~9.09~1316~9.10~2574~9.11~5680~9.12~1234~~'
    '20260918150003~0.01~0.11~9.15~9.00~9.07/517593/469320000~517593~46932~0.35~5.91";'
)


def test_star_market_volume_is_already_shares():
    """科创板成交量字段本来就是"股"，不能再 ×100（本文件存在的理由）。"""
    row = tq.parse_quotes(REAL)["sh688525"]
    assert row["volume"] == 18_893_455, "688 原始值就是股数（== 新浪同日股数），×100 会放大 100 倍"
    assert row["amount"] == 4_076_690_406       # 真实成交额（元），不是 volume×close 估算
    assert row["close"] == 216.88 and row["open"] == 215.00
    assert str(row["date"].date()) == "2026-09-18"


def test_main_board_volume_is_hand_and_multiplied_by_100():
    """非科创板仍是"手"→×100 得股（不能因为修 688 而把这条一起改坏）。"""
    row = tq.parse_quotes(REAL)["sh600000"]
    assert row["volume"] == 517_593 * 100
    assert row["amount"] == 469_320_000


def test_volume_matches_amount_over_close():
    """行内自洽：volume ≈ amount/close（对全部板块都该成立，是审计脚本的判据）。"""
    for code in ("sh688525", "sh600000"):
        row = tq.parse_quotes(REAL)[code]
        ratio = row["volume"] * row["close"] / row["amount"]
        assert 0.9 < ratio < 1.1, f"{code} volume 与 amount/close 不自洽：ratio={ratio:.3f}"


def test_suspended_and_malformed_rows_are_dropped():
    """停牌（价格/量为 0）与残缺行必须丢弃，交回个股源链按 no_data 处理。"""
    suspended = ('v_sh600001="1~某某~600001~0.00~0.00~0.00~0~0~0~~0~0~0~0~0~0~0~0~0~0~0~'
                 '0~0~0~0~0~0~~20260918150000~0~0~0~0~0.00/0/0~0~0";')
    assert tq.parse_quotes(suspended) == {}
    assert tq.parse_quotes("v_sh600002=\"1~残缺\";") == {}
    assert tq.parse_quotes("") == {}


def test_fetch_quote_bars_builds_single_row_frames(monkeypatch):
    """分块取数 → 单行 DataFrame（date 索引），停牌股不出现（由调用方走源链）。"""
    class _Resp:
        status_code = 200
        encoding = "gbk"
        text = REAL

        def __init__(self):
            self.encoding = "gbk"

    class _Session:
        def get(self, url, timeout=None):        # noqa: ARG002
            assert "sh688525" in url and "sh600000" in url
            return type("R", (), {"status_code": 200, "encoding": "gbk", "text": REAL})()

    monkeypatch.setattr(tq, "_session", lambda: _Session())
    bars = tq.fetch_quote_bars(["688525", "600000", "000001"], batch_size=3)
    assert set(bars) == {"688525", "600000"}     # 000001 不在响应里 → 不在结果里
    assert bars["688525"].iloc[0]["volume"] == 18_893_455
    assert list(bars["688525"].columns) == ["open", "high", "low", "close", "volume", "amount"]
