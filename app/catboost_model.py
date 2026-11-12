"""Lazy-imported CatBoost adapters with fold-local categorical feature discovery.

Model parameters keep their native sklearn paths. Missing/category string encoding
is performed by model_commit.prepare_predictors before these estimators see X.
"""
from catboost import CatBoostClassifier, CatBoostRegressor


def _categorical_features(X):
    return list(X.select_dtypes(include=['category', 'object', 'string', 'bool']).columns)


class PreparedCatBoostRegressor(CatBoostRegressor):
    def fit(self, X, y=None, **kwargs):
        kwargs.setdefault('cat_features', _categorical_features(X))
        return super().fit(X, y, **kwargs)


class PreparedCatBoostClassifier(CatBoostClassifier):
    def fit(self, X, y=None, **kwargs):
        kwargs.setdefault('cat_features', _categorical_features(X))
        return super().fit(X, y, **kwargs)
