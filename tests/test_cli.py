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


def _explode(args):
    raise RuntimeError("模拟数据源不可达")


def test_main_turns_uncaught_exception_into_readable_failure(monkeypatch, capsys):
    """顶层兜底：子命令抛异常时必须给出"失败说明 + 完整堆栈"，而不是裸 traceback。

    背景（2026-09-23 实测）：全新克隆后跑 `daily` 撞到数据源不可达，用户看到的
    是一屏 requests 栈，没有上下文、也没有下一步。退出码约定不变（1 = 未捕获异常）。

    还原旧行为（去掉 main() 里的 try/except）时本测试必红。
    """
    import ashare_quant.cli as cli

    # 让 daily 子命令指向会抛异常的桩函数
    real_parse = cli.argparse.ArgumentParser.parse_args

    def fake_parse(self, argv=None):
        ns = real_parse(self, argv)
        ns.func = _explode
        return ns

    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", fake_parse)

    try:
        main(["daily"])
        raise AssertionError("main() 应当以 SystemExit 结束")
    except SystemExit as e:
        assert e.code == 1, f"未捕获异常应返回退出码 1，实际 {e.code}"

    err = capsys.readouterr().err
    assert "运行失败" in err, "必须打印可读的失败说明"
    assert "RuntimeError" in err and "模拟数据源不可达" in err, "必须带上异常类型与消息"
    assert "常见原因" in err, "必须给出常见原因（新用户需要下一步指引）"
    assert "fetch" in err, "必须指向'先下载数据'这条最常见的下一步"
    assert "Traceback (most recent call last)" in err, (
        "完整堆栈必须保留 —— 藏掉堆栈会让 bug 报告变难")


def test_main_keyboard_interrupt_exits_130(monkeypatch, capsys):
    """Ctrl-C 用 130（128+SIGINT）退出，且不打印 traceback。"""
    import ashare_quant.cli as cli

    def boom(args):
        raise KeyboardInterrupt

    real_parse = cli.argparse.ArgumentParser.parse_args

    def fake_parse(self, argv=None):
        ns = real_parse(self, argv)
        ns.func = boom
        return ns

    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", fake_parse)

    try:
        main(["daily"])
        raise AssertionError("main() 应当以 SystemExit 结束")
    except SystemExit as e:
        assert e.code == 130
    assert "Traceback" not in capsys.readouterr().err


def test_main_still_propagates_data_failure_exit_code(monkeypatch):
    """子命令主动返回 2（数据侧故障）时必须原样透出 —— 别被顶层兜底吞掉。"""
    import ashare_quant.cli as cli

    real_parse = cli.argparse.ArgumentParser.parse_args

    def fake_parse(self, argv=None):
        ns = real_parse(self, argv)
        ns.func = lambda args: cli.EXIT_DATA_FAILURE
        return ns

    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", fake_parse)

    try:
        main(["daily"])
        raise AssertionError("main() 应当以 SystemExit 结束")
    except SystemExit as e:
        assert e.code == 2

