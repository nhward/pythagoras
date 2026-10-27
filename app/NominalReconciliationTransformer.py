"""Replay human-approved nominal mappings without learning from validation data."""
from copy import deepcopy

import numpy as np
import pandas as pd
from code_recording import recordable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@recordable
def apply_merges(frame, items):
    result = frame.copy()
    for column in dict.fromkeys(p['column'] for p in items):
        mapping = {}
        for p in items:
            if p['column'] != column:
                continue
            for label in p['sources']:
                if label in mapping and mapping[label] != p['destination']:
                    raise ValueError(f"Conflicting destinations for {label}")
                mapping[label] = p['destination']
        def resolve(label, mapping):
            seen = set()
            while label in mapping:
                if label in seen:
                    raise ValueError("Merge decisions contain a cycle")
                seen.add(label)
                label = mapping[label]
            return label
        mapping = {key: resolve(key, mapping) for key in mapping}
        series = result[column]
        if not isinstance(series.dtype, pd.CategoricalDtype):
            if not all(isinstance(v, str) for v in series.dropna()):
                raise ValueError(f'{column!r} must contain nominal string labels.')
            series = pd.Series(pd.Categorical(series), index=series.index)
        if series.cat.ordered or not all(isinstance(v, str) for v in series.cat.categories):
            raise ValueError(f'{column!r} must contain unordered nominal string labels.')
        categories = list(dict.fromkeys(mapping.get(c, c) for c in series.cat.categories))
        values = [v if pd.isna(v) else mapping.get(v, v) for v in series.array]
        result[column] = pd.Categorical(values, categories=categories, ordered=False)
    return result


@recordable
class NominalReconciliationTransformer(TransformerMixin, BaseEstimator):
    """Fixed mappings, simultaneous within each batch and sequential across batches.

    Unmapped labels and missing values pass through. No frequencies, vocabulary,
    or similarity decisions are learned by fit.
    """

    def __init__(self, batches):
        self.batches = batches

    @property
    def columns(self):
        return tuple(dict.fromkeys(p['column'] for batch in self.batches for p in batch))

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Nominal reconciliation requires a DataFrame with unique columns.')
        absent = [c for c in self.columns if c not in X]
        if absent:
            raise ValueError(f'Missing nominal columns: {absent}')

    def fit(self, X, y=None):
        self._validate(X)
        # Validate mappings even if their source labels are absent in this fold.
        preview = X.iloc[:0]
        for batch in self.batches:
            preview = apply_merges(preview, batch)
        self.batches_ = deepcopy(self.batches)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, 'batches_')
        self._validate(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Nominal reconciliation columns must match the fitted schema.')
        result = X.copy()
        for batch in self.batches_:
            result = apply_merges(result, batch)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError('input_features must match the fitted schema.')
        return self.feature_names_in_.copy()
