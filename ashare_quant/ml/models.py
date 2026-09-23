from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_REGISTRY: dict[str, Callable] = {}
_PLUGINS_LOADED = False


def register_model(name: str | None = None):
    """注册一个模型工厂；用于内置模型与第三方插件。

    usage:
        @register_model("my_model")
        def my_model():
            return SomeEstimator()
    """
    def deco(factory: Callable) -> Callable:
        key = name or factory.__name__
        if key in _REGISTRY:
            raise ValueError(f"model already registered: {key}")
        _REGISTRY[key] = factory
        return factory
    return deco


def _load_plugins() -> None:
    """从 entry_points(group='ashare_quant.models') 发现第三方模型插件。"""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        eps = entry_points()
        group = eps.select(group="ashare_quant.models") if hasattr(eps, "select") \
            else eps.get("ashare_quant.models", [])
        for ep in group:
            try:
                factory = ep.load()
            except Exception:  # noqa: BLE001 单个插件失败不影响主流程
                continue
            _REGISTRY.setdefault(ep.name, factory)
    except Exception:  # noqa: BLE001
        return


@register_model()
def linear() -> Ridge:
    return Ridge(alpha=1e-6, solver="sparse_cg", random_state=0)


@register_model()
def rf() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=0)


@register_model()
def lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         n_jobs=-1, random_state=0, verbose=-1)


@register_model()
def xgb():
    """XGBoost：与 LGBM/HistGB 同族不同实现，给集成加多样性。"""
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                        n_jobs=-1, random_state=0, verbosity=0)


class _LGBMRankerWrapper:
    """LGBMRanker（LambdaRank）：直接优化横截面排序（按日期分组）。

    我们的任务是"预测未来 20 日收益后排序选 Top-N"，MSE 回归对排序
    是代理目标；LGBMRanker 用 pairwise 排序损失直接对齐任务目标，
    walk-forward 评测中每折独立 fit（无需 sklearn clone）。
    """

    def __init__(self, **kwargs) -> None:
        self._kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                        n_jobs=-1, random_state=0, verbose=-1,
                        label_gain=list(range(10)))
        self._kw.update(kwargs)
        self.model = None

    def fit(self, X, y, **fit_kwargs):
        from lightgbm import LGBMRanker
        dates = X.index.get_level_values("date")
        order = np.argsort(dates.to_numpy(), kind="stable")
        Xs = X.iloc[order]
        ys = y.iloc[order]
        ds = dates[order]
        groups = ds.value_counts().sort_index().to_numpy()
        # LambdaRank 需要整数 relevance：把连续收益转成"日期内十分位"（0-9）
        label = (ys.groupby(level="date").rank(pct=True) * 9).round().astype(int)
        self.model = LGBMRanker(**self._kw)
        self.model.fit(Xs, label, group=groups, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)


@register_model()
def rank_lgb():
    """LightGBM 排序（LambdaRank），按日期分组优化横截面排名。"""
    return _LGBMRankerWrapper()


class _XGBRankerWrapper:
    """XGBoost ranker（pairwise）：与 rank_lgb 同思路，排序目标异构补充。"""

    def __init__(self, **kwargs) -> None:
        self._kw = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                        n_jobs=-1, random_state=0, verbosity=0)
        self._kw.update(kwargs)
        self.model = None

    def fit(self, X, y, **fit_kwargs):
        from xgboost import XGBRanker
        dates = X.index.get_level_values("date")
        order = np.argsort(dates.to_numpy(), kind="stable")
        Xs = X.iloc[order]
        ys = y.iloc[order]
        ds = dates[order]
        groups = ds.value_counts().sort_index().to_numpy()
        label = (ys.groupby(level="date").rank(pct=True) * 9).round().astype(int)
        self.model = XGBRanker(**self._kw)
        self.model.fit(Xs, label, group=groups, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)


@register_model()
def rank_xgb():
    """XGBoost 排序（pairwise），按日期分组优化横截面排名。"""
    return _XGBRankerWrapper()


@register_model()
def mlp_deep():
    """更深 MLP（带早停）：捕捉特征非线性交互，比默认 (64,32) 更强。"""
    return MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                        early_stopping=True, n_iter_no_change=20,
                        random_state=0)


class _PLS:
    """偏最小二乘（Gu-Kelly-Xiu 大样本计量）：成分数自适应特征数。"""

    def __init__(self, n_components: int = 8) -> None:
        self.n_components = n_components
        self.model = None

    def fit(self, X, y, **fit_kwargs):
        from sklearn.cross_decomposition import PLSRegression
        k = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self.model = PLSRegression(n_components=k)
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)


@register_model()
def pls():
    """偏最小二乘（Gu-Kelly-Xiu 大样本计量）：从特征中提取主成分预测。"""
    return _PLS()


@register_model()
def enet():
    """弹性网：稀疏线性 + 岭，特征多时比 Ridge 更稳。"""
    from sklearn.linear_model import ElasticNet
    return ElasticNet(alpha=1e-3, l1_ratio=0.5, random_state=0)


@register_model()
def huber_lgb():
    """LGBM + Huber 损失：对收益厚尾/异常值更稳健。"""
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         objective="huber", n_jobs=-1, random_state=0, verbose=-1)


class _TemporalDecayLGB:
    """自研：时间衰减 LGBM——近期样本权重更高，适应市场状态漂移。

    权重 = exp(-天数 / 365)，约一年前样本权重降到 37%，半衰期一年。
    """

    def __init__(self, half_life_days: int = 365, **kwargs) -> None:
        self._half = half_life_days
        self._kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                        n_jobs=-1, random_state=0, verbose=-1)
        self._kw.update(kwargs)
        self.model = None

    def fit(self, X, y, **fit_kwargs):
        from lightgbm import LGBMRegressor
        dates = X.index.get_level_values("date")
        age = (dates.max() - dates).days.to_numpy(dtype=float)
        w = np.exp(-np.log(2) * age / self._half)
        self.model = LGBMRegressor(**self._kw)
        self.model.fit(X, y, sample_weight=w, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)


@register_model()
def temporal_decay_lgb():
    """自研：时间衰减 LGBM（近期样本加权，半衰期一年）。"""
    return _TemporalDecayLGB()


class _RiskAwareLGB:
    """自研：风险调整 LGBM——双头模型。

    头1 预测未来收益，头2 预测未来收益绝对值（风险代理）；
    打分 = 预期收益 / (1 + 预期风险)，选"单位风险收益高"的股票。
    """

    def __init__(self, **kwargs) -> None:
        self._kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                        n_jobs=-1, random_state=0, verbose=-1)
        self._kw.update(kwargs)
        self.ret_model = None
        self.risk_model = None

    def fit(self, X, y, **fit_kwargs):
        from lightgbm import LGBMRegressor
        self.ret_model = LGBMRegressor(**self._kw)
        self.ret_model.fit(X, y, **fit_kwargs)
        self.risk_model = LGBMRegressor(**self._kw)
        self.risk_model.fit(X, y.abs(), **fit_kwargs)
        return self

    def predict(self, X):
        ret = self.ret_model.predict(X)
        risk = self.risk_model.predict(X)
        return ret / (1.0 + np.maximum(risk, 0.0))


@register_model()
def risk_aware_lgb():
    """自研：风险调整 LGBM（收益头 ÷ (1+风险头)）。"""
    return _RiskAwareLGB()


@register_model()
def histgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                         learning_rate=0.05, random_state=0)


@register_model()
def svm():
    return make_pipeline(StandardScaler(), SVR(C=1.0))


@register_model()
def knn():
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=20, n_jobs=-1))


@register_model()
def mlp():
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32),
                                                        max_iter=300, random_state=0))


_load_plugins()

# 向后兼容：MODELS 指向注册表（dict 语义不变，含插件模型）
MODELS = _REGISTRY


def list_models() -> list[str]:
    """按注册顺序列出所有可用模型名。"""
    return list(_REGISTRY)


def make_model(name: str):
    """按名字构造模型实例，未注册时给出清晰报错。"""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}', available: {list_models()}")
    return _REGISTRY[name]()
