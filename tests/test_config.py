from pathlib import Path
from ashare_quant.config import Config


def test_defaults_when_yaml_missing(tmp_path):
    cfg = Config.from_yaml(tmp_path / "nope.yaml")
    assert cfg.universe_mode == "csi300"
    assert cfg.years == 3
    assert cfg.adjust == "qfq"


def test_from_dict_overrides():
    cfg = Config.from_dict({"years": 5, "universe_mode": "all"})
    assert cfg.years == 5
    assert cfg.universe_mode == "all"
