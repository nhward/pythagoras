"""Sklearn-compatible boolean to nullable integer encoding."""
from __future__ import annotations

import numpy as np
import pandas as pd
from code_recording import recordable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@recordable
class LogicalEncodingTransformer(TransformerMixin, BaseEstimator):
    """Encode False as 0 and True as 1, preserving missing observations."""

    def __init__(self, columns, *, remove_original=True):
        self.columns = columns
        self.remove_original = remove_original

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Logical encoding requires a DataFrame with unique column names.')
        absent = [c for c in self.columns if c not in X]
        if absent:
            raise ValueError(f'Missing Logical predictor columns: {absent}')

    def fit(self, X, y=None):
        self._validate(X)
        self.encodings_ = {}
        self.output_columns_ = []
        self.removed_columns_ = []
        self.summary_ = []
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        used = set(X.columns)
        for column in dict.fromkeys(self.columns):
            series = X[column]
            if not pd.api.types.is_bool_dtype(series.dtype):
                raise ValueError(f'{column!r} needs a Logical (boolean) dtype during fitting.')
            base = f'{column}__integer'
            name, number = base, 2
            while name in used:
                name = f'{base}_{number}'
                number += 1
            used.add(name)
            self.encodings_[column] = name
            self.output_columns_.append(name)
            if self.remove_original:
                self.removed_columns_.append(column)
            self.summary_.append({'Variable': str(column), 'False': int(series.eq(False).sum()),
                'True': int(series.eq(True).sum()), 'Missing': int(series.isna().sum()),
                'Mapping': 'False → 0; True → 1', 'New variable': name})
        self.output_features_ = [c for c in X if c not in self.removed_columns_] + self.output_columns_
        return self

    def transform(self, X):
        check_is_fitted(self, 'encodings_')
        self._validate(X)
        if set(self.output_columns_) & set(X.columns):
            raise ValueError('A logical output name already exists in the incoming data.')
        out = X.drop(columns=self.removed_columns_).copy()
        for column, name in self.encodings_.items():
            series = X[column]
            # Never use truthiness: strings such as "False" would become True.
            if not pd.api.types.is_bool_dtype(series.dtype) and not all(
                    isinstance(value, (bool, np.bool_)) for value in series.dropna()):
                raise ValueError(f'{column!r} must contain boolean values or missing values.')
            out[name] = pd.array(series, dtype='boolean').astype('Int64')
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'output_features_')
        return np.asarray(self.output_features_, dtype=object)
