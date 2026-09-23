import pandas as pd
from ashare_quant.universe import filter_universe


def test_filter_universe_drops_st():
    raw = pd.DataFrame({"code": ["000001", "000002", "600001"], "name": ["平安银行", "*ST海投", "华夏银行"]})
    out = filter_universe(raw)
    assert out["code"].tolist() == ["000001", "600001"]


def test_filter_universe_keeps_all_when_disabled():
    raw = pd.DataFrame({"code": ["000001"], "name": ["*ST海投"]})
    out = filter_universe(raw, drop_st=False)
    assert len(out) == 1
