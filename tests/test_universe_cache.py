import json

from ashare_quant.universe import load_universe_cached


def test_load_universe_cached_writes_and_reuses(tmp_path, monkeypatch):
    cache = tmp_path / "universe.json"
    calls = []

    def fake_load(mode):
        calls.append(mode)
        return ["000001", "000002"]

    monkeypatch.setattr("ashare_quant.universe.load_universe", fake_load)

    first = load_universe_cached("all", cache_path=cache, max_age_days=7)
    assert first == ["000001", "000002"]
    assert calls == ["all"]
    assert cache.exists()

    # 第二次应直接读缓存，不触发联网
    second = load_universe_cached("all", cache_path=cache, max_age_days=7)
    assert second == ["000001", "000002"]
    assert calls == ["all"]

    # extra 并集
    merged = load_universe_cached("all", cache_path=cache, max_age_days=7,
                                  extra=["000003"])
    assert merged == ["000001", "000002", "000003"]


def test_load_universe_cached_merges_extra_on_refresh(tmp_path, monkeypatch):
    cache = tmp_path / "universe.json"
    cache.write_text(json.dumps({
        "mode": "all",
        "fetched_at": "2000-01-01",
        "codes": ["000001"],
    }), encoding="utf-8")
    calls = []

    def fake_load(mode):
        calls.append(mode)
        return ["000001", "000002"]

    monkeypatch.setattr("ashare_quant.universe.load_universe", fake_load)
    codes = load_universe_cached("all", cache_path=cache, max_age_days=7)
    assert calls == ["all"]  # 过期缓存触发刷新
    assert codes == ["000001", "000002"]
