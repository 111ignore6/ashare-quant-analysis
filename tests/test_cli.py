from ashare_quant.cli import main


def test_main_fetch_help():
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0


def test_main_daily_requires_no_args():
    try:
        main(["daily", "--help"])
    except SystemExit as e:
        assert e.code == 0
