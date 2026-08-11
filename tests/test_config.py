from ashare_quant.config import Config, update_config_yaml


def test_defaults_when_yaml_missing(tmp_path):
    cfg = Config.from_yaml(tmp_path / "nope.yaml")
    assert cfg.universe_mode == "csi300"
    assert cfg.years == 3
    assert cfg.adjust == "qfq"


def test_from_dict_overrides():
    cfg = Config.from_dict({"years": 5, "universe_mode": "all"})
    assert cfg.years == 5
    assert cfg.universe_mode == "all"


def test_to_dict_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("initial_capital: 100000.0\ntop_n: 50\nstop_loss: -0.15\n",
                 encoding="utf-8")
    cfg = Config.from_yaml(p)
    d = cfg.to_dict()
    assert d["data_root"] == "data"
    assert d["initial_capital"] == 100000.0


def test_update_config_yaml_preserves_others(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "universe_mode: all\nyears: 3\ndata_source: tencent\n"
        "initial_capital: 100000.0\ntop_n: 50\nstop_loss: -0.15\n"
        "take_profit: 0.3\nauto_update: true\n",
        encoding="utf-8")
    merged = update_config_yaml(
        p,
        initial_capital=250000,
        top_n=80,
        stop_loss=None,
        data_source="akshare",
    )
    assert merged["initial_capital"] == 250000.0
    assert merged["top_n"] == 80
    assert merged["stop_loss"] is None
    assert merged["data_source"] == "akshare"
    assert merged["universe_mode"] == "all"   # 未提及字段保留
    assert merged["auto_update"] is True
    reloaded = Config.from_yaml(p)
    assert reloaded.initial_capital == 250000.0
    assert reloaded.top_n == 80
    assert reloaded.stop_loss is None
    assert reloaded.data_source == "akshare"
