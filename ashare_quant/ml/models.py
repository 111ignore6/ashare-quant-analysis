from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def linear() -> Ridge:
    return Ridge(alpha=1e-6, solver="sparse_cg", random_state=0)


def rf() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=0)


def lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         n_jobs=-1, random_state=0, verbose=-1)


def histgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                         learning_rate=0.05, random_state=0)


def svm():
    return make_pipeline(StandardScaler(), SVR(C=1.0))


def knn():
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=20, n_jobs=-1))


def mlp():
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32),
                                                        max_iter=300, random_state=0))


MODELS = {
    "linear": linear,
    "rf": rf,
    "lgbm": lgbm,
    "histgb": histgb,
    "svm": svm,
    "knn": knn,
    "mlp": mlp,
}
