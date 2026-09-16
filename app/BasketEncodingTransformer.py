"""Training-vocabulary binary encoding for Basket predictors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from list_pandas import is_list
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _items(value):
    """Return distinct nonmissing scalar items, or None for a missing basket."""
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    if not isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        raise ValueError('Basket values must be collections of scalar items, or missing.')
    items = {}
    for item in value:
        if not pd.api.types.is_scalar(item):
            raise ValueError('Basket items must be hashable scalar values; nested collections are unsupported.')
        if pd.isna(item):
            continue
        try:
            items[item] = None
        except TypeError as error:
            raise ValueError('Basket items must be hashable scalar values.') from error
    return items


class BasketEncodingTransformer(TransformerMixin, BaseEstimator):
    """One binary feature per fitted item; ignore unknown items at transform time.

    Duplicates are ignored. Empty baskets encode as zeros; missing baskets encode
    as missing in every output. No reference item is dropped: items can coexist.
    """

    def __init__(self, columns, *, remove_original=True):
        self.columns = columns
        self.remove_original = remove_original

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Basket encoding requires a DataFrame with unique column names.')
        absent = [c for c in self.columns if c not in X]
        if absent:
            raise ValueError(f'Missing Basket predictor columns: {absent}')

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
            if not is_list(X[column]):
                raise ValueError(f'{column!r} needs a Basket dtype during fitting.')
            vocabulary = {}
            missing = empty = 0
            for value in X[column]:
                items = _items(value)
                if items is None:
                    missing += 1
                    continue
                empty += not items
                for item in items:
                    if item not in vocabulary:
                        if len(self.output_columns_) + len(vocabulary) >= 4096:
                            raise ValueError('Basket encoding exceeds 4096 indicators. Reduce the basket vocabulary upstream.')
                        vocabulary[item] = len(vocabulary)
            names = []
            for item in vocabulary:
                base = f'{column}__{item}'
                name, suffix = base, 2
                while name in used:
                    name = f'{base}_{suffix}'
                    suffix += 1
                used.add(name)
                names.append(name)
            self.encodings_[column] = (vocabulary, names)
            self.output_columns_.extend(names)
            if names and self.remove_original:
                self.removed_columns_.append(column)
            self.summary_.append({'Variable': str(column), 'Distinct items': len(vocabulary),
                'Items': ', '.join(map(str, vocabulary)), 'Missing baskets': missing,
                'Empty baskets': empty, 'Outputs': len(names), 'New variables': ', '.join(names),
                'Status': 'Binary presence; duplicates and unseen items ignored.' if names else
                    'No observed items; original retained.'})
        self.output_features_ = [c for c in X if c not in self.removed_columns_] + self.output_columns_
        return self

    def transform(self, X):
        check_is_fitted(self, 'encodings_')
        self._validate(X)
        if set(self.output_columns_) & set(X.columns):
            raise ValueError('A basket output name already exists in the incoming data.')
        parts = [X.drop(columns=self.removed_columns_).copy()]
        for column, (vocabulary, names) in self.encodings_.items():
            values = np.zeros((len(X), len(names)), dtype=np.int8)
            missing = np.zeros(len(X), dtype=bool)
            for row, value in enumerate(X[column]):
                items = _items(value)
                if items is None:
                    missing[row] = True
                    continue
                for item in items:
                    position = vocabulary.get(item)
                    if position is not None:
                        values[row, position] = 1
            encoded = pd.DataFrame(values, index=X.index, columns=names, dtype='Int8')
            encoded.loc[missing, :] = pd.NA
            parts.append(encoded)
        return pd.concat(parts, axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'output_features_')
        return np.asarray(self.output_features_, dtype=object)
