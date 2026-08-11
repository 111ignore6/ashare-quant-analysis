"""仪表盘冒烟测试：mock 行情后整页运行，确保无异常（回归保护接线 bug）。"""

from pathlib import Path


PROJ = Path(__file__).resolve().parents[1]


class _FakeEQ:
    """模拟行情源：指数 + 个股快照均返回稳定数据。"""

    def stocks(self, codes):
        out = {}
        for c in codes:
            if c in ("sh000001", "sz399001", "sz399006", "sh000300"):
                out[c] = {"name": c, "now": 3000.0, "close": 2980.0}
            else:
                out[c] = {"name": "测试股", "now": 10.0, "close": 9.9,
                          "open": 9.8, "high": 10.1, "low": 9.7, "volume": 100}
        return out


def test_dashboard_smoke(monkeypatch):
    monkeypatch.setattr("easyquotation.use", lambda source: _FakeEQ())
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJ / "dashboard.py"), default_timeout=180)
    at.run(timeout=180)
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) >= 8


def test_dashboard_smoke_cross_refresh(monkeypatch):
    """跨刷新缓存路径回归：st.cache_data 嵌套曾导致 KeyError，连续运行必须稳定。"""
    monkeypatch.setattr("easyquotation.use", lambda source: _FakeEQ())
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJ / "dashboard.py"), default_timeout=180)
    for _ in range(3):
        at.run(timeout=180)
        assert not at.exception, [e.value for e in at.exception]
