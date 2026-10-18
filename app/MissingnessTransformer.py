"""Fold-fitted column selection and target-aligned missing-row removal."""

import numpy as np
import pandas as pd
from code_recording import recordable
from feature_engine.imputation import DropMissingData
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@recordable
class MissingnessTransformer(TransformerMixin, BaseEstimator):
    """Learn excessive-missingness columns, then filter rows on retained columns."""

    def __init__(self, *, remove_variables=False, remove_observations=False,
                 variable_threshold=0.5, observation_threshold=0.5):
        self.remove_variables = remove_variables
        self.remove_observations = remove_observations
        self.variable_threshold = variable_threshold
        self.observation_threshold = observation_threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("Missingness removal requires a DataFrame with unique columns")
        for threshold in (self.variable_threshold, self.observation_threshold):
            if not 0 <= threshold <= 1:
                raise ValueError("Missingness thresholds must be between zero and one")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.removed_columns_ = (
            list(X.columns[X.isna().mean() > self.variable_threshold])
            if self.remove_variables else []
        )
        self.columns_ = [c for c in X if c not in self.removed_columns_]
        self.row_filter_ = None
        count = len(self.columns_)
        if self.remove_observations and count and len(X) and self.observation_threshold < 1:
            # Convert the strict missingness boundary to an integer count first.
            # nextafter avoids rounding an exact count slightly upwards in dropna.
            allowed = sum(k / count <= self.observation_threshold for k in range(count + 1)) - 1
            threshold = float(np.nextafter((count - allowed) / count, 0.0))
            self.row_filter_ = DropMissingData(
                threshold=threshold, variables=self.columns_, missing_only=False,
            ).fit(X.loc[:, self.columns_])
        return self

    def _transform_with_positions(self, X):
        check_is_fitted(self, "columns_")
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Missingness removal requires the fitted input columns in order")
        frame = X.loc[:, self.columns_].copy()
        positions = np.arange(len(frame))
        if self.row_filter_ is not None and len(frame):
            # Position-based selection also works with duplicate source labels.
            positions = self.row_filter_.transform(frame.reset_index(drop=True)).index.to_numpy()
        return frame.iloc[positions].copy(), positions

    def transform(self, X):
        return self._transform_with_positions(X)[0]

    def transform_x_y(self, X, y):
        frame, positions = self._transform_with_positions(X)
        if y is None:
            return frame, None
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of rows")
        target = y.iloc[positions].copy() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y)[positions]
        return frame, target

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")
        return np.asarray(self.columns_, dtype=object)
