import importlib
import pathlib

import numpy as np
import pandas as pd

from ashare_quant.research.stats import distribution_stats, volatility_clustering


def _preload_statsmodels_stattools() -> None:
    """收集期显式预加载 statsmodels 的编译扩展（让偶发失败变响亮、可诊断）。

    背景见 AGENTS.md「已知问题」1：`statsmodels.tsa.stattools` 只在 `acorr_ljungbox()`
    内部被延迟导入，其中第 39 行加载编译扩展
    `statsmodels/tsa/_innovations.cp313-win_amd64.pyd`。Windows 上该扩展偶发首次加载失败
    （2026-09-16 全量测试出现过 1 次），失败点埋在 site-packages 深处，光看 traceback
    很难判断是环境抖动还是装坏了。这里在**收集期**提前加载：

    - 失败 ⇒ 本文件收集失败，整轮测试直接变红（而不是测试体深处一个 ImportError）；
    - 异常消息补上 winerror 与磁盘上实际的 .pyd 路径，便于一眼分类。

    原则：**不吞异常、不放宽任何断言**——加载不了就是红。
    """
    try:
        importlib.import_module("statsmodels.tsa.stattools")
    except ImportError as exc:
        import statsmodels

        pkg = pathlib.Path(statsmodels.__file__).parent
        found = sorted(str(p.relative_to(pkg)) for p in pkg.rglob("_innovations*.pyd"))
        raise ImportError(
            f"statsmodels 编译扩展首次加载失败：{exc!r}；"
            f"winerror={getattr(exc, 'winerror', None)}；"
            f"已安装的 _innovations 扩展={found or '未找到（安装损坏？）'}；"
            "若重跑一次即绿，见 AGENTS.md「已知问题」1（环境级偶发，未能复现）"
        ) from exc


_preload_statsmodels_stattools()


def _close_panel(n_days=1000, n_stocks=50, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    cols = {f"S{i:04d}": 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days))) for i in range(n_stocks)}
    return pd.DataFrame(cols, index=idx)


def test_distribution_stats_reports_kurtosis():
    close = _close_panel()
    dist = distribution_stats(close)
    assert "kurtosis" in dist
    assert dist["skewness"] is not None
    assert dist["n_symbols"] == 50


def test_volatility_clustering_on_garch_like_series():
    rng = np.random.default_rng(7)
    n = 2000
    ret = np.zeros(n)
    sigma2 = np.ones(n) * 1e-4
    for t in range(1, n):
        sigma2[t] = 1e-5 + 0.9 * ret[t - 1] ** 2 + 0.08 * sigma2[t - 1]
        ret[t] = rng.normal(0, np.sqrt(sigma2[t]))
    out = volatility_clustering(pd.Series(ret))
    assert out["ljungbox_p"] < 0.05
    assert out["arch_p"] < 0.05
