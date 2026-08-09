import pandas as pd
from ashare_quant.cache import ParquetStore
from ashare_quant.config import Config
from ashare_quant.pipeline import build_panels, download_universe


def _df(dates, close):
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": close, "high": [c + 0.1 for c in close], "low": [c - 0.1 for c in close],
         "close": close, "volume": [1000] * len(idx), "amount": [1e6] * len(idx)},
        index=idx,
    )


def test_download_universe_with_fake_fetcher(tmp_path):
    store = ParquetStore(tmp_path)
    cfg = Config.from_dict({"years": 1, "retry": 1})
    calls: list[str] = []

    def fake_fetcher(code, start, end, adjust):
        calls.append(code)
        return _df(["2024-01-02", "2024-01-03"], [10, 10.5])

    res = download_universe(["000001", "000002"], store, cfg, fetcher=fake_fetcher)
    assert set(calls) == {"000001", "000002"}
    assert res["ok"] == ["000001", "000002"]
    assert store.exists("000001")


def test_download_universe_skips_existing(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02"], [10]))
    cfg = Config.from_dict({"years": 1, "retry": 1})
    res = download_universe(["000001"], store, cfg, fetcher=lambda *a, **k: _df(["2024-01-03"], [11]))
    assert res["skipped"] == ["000001"]


def test_build_panels(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02", "2024-01-03"], [10, 10.5]))
    store.save("000002", _df(["2024-01-02", "2024-01-03"], [20, 19]))
    panels = build_panels(store)
    assert panels["close"].shape == (2, 2)
    assert panels["close"].columns.tolist() == ["000001", "000002"]


def test_download_universe_progress(capsys, tmp_path):
    store = ParquetStore(tmp_path)
    cfg = Config.from_dict({"years": 1, "retry": 1})

    def fake_fetcher(code, start, end, adjust):
        return _df(["2024-01-02"], [10])

    download_universe(["000001", "000002", "000003"], store, cfg,
                      fetcher=fake_fetcher, progress_every=1)
    captured = capsys.readouterr().out
    assert "progress 3/3" in captured
