import pandas as pd
from ashare_quant.cache import ParquetStore


def _df(dates, offset=0.0):
    idx = pd.to_datetime(dates)
    n = len(idx)
    return pd.DataFrame(
        {"open": [10 + offset] * n, "high": [11 + offset] * n, "low": [9 + offset] * n,
         "close": [10.5 + offset] * n, "volume": [1000] * n, "amount": [1e6] * n},
        index=idx,
    )


def test_roundtrip(tmp_path):
    store = ParquetStore(tmp_path)
    store.save("000001", _df(["2024-01-02", "2024-01-03"]))
    loaded = store.load("000001")
    assert list(loaded.index) == pd.to_datetime(["2024-01-02", "2024-01-03"]).tolist()
    assert store.symbols() == ["000001"]


def test_append_dedups(tmp_path):
    store = ParquetStore(tmp_path)
    store.append("000001", _df(["2024-01-02", "2024-01-03"]))
    store.append("000001", _df(["2024-01-03", "2024-01-04"]))
    loaded = store.load("000001")
    assert len(loaded) == 3
    m = store.read_manifest()
    assert m["000001"]["rows"] == 3


def test_missing_symbol_returns_none(tmp_path):
    store = ParquetStore(tmp_path)
    assert store.load("999999") is None
    assert not store.exists("999999")


def test_concurrent_saves_keep_complete_manifest(tmp_path):
    import threading

    store = ParquetStore(tmp_path)
    symbols = [f"{i:06d}" for i in range(30)]
    threads = [threading.Thread(target=store.save, args=(s, _df(["2024-01-02"])))
               for s in symbols]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    m = store.read_manifest()
    assert len(m) == len(symbols)
