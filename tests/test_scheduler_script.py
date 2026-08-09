from pathlib import Path


def test_schedule_script_exists():
    p = Path("scripts/schedule_daily.ps1")
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "AshareQuantDaily" in text
    assert "daily" in text
