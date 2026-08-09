from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

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
