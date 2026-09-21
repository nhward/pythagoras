"""Train-only one-hot encoding with DataFrame and missing-value preservation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from code_recording import recordable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


@recordable
class NominalEncodingTransformer(TransformerMixin, BaseEstimator):
    """Encode explicitly selected nominal predictors using only fitted levels.

    Category discovery ignores pandas' unused category metadata. Missing rows
    stay missing in every indicator. Unseen values follow ``handle_unknown``.
    Integer tokens let sklearn handle even mixed-type categorical labels without
    merging labels with identical text representations.
    """

    def __init__(self, columns, *, remove_original=True, min_frequency=None,
                 max_categories=None, handle_unknown='ignore'):
        self.columns = columns
        self.remove_original = remove_original
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.handle_unknown = handle_unknown

    def _validate_frame(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Nominal encoding requires a DataFrame with unique column names.')
        missing = [c for c in self.columns if c not in X]
        if missing:
            raise ValueError(f'Missing encoding columns: {missing}')

    def fit(self, X, y=None):
        self._validate_frame(X)
        if not len(X):
            raise ValueError('Nominal encoding needs at least one observation.')
        if self.handle_unknown not in ('ignore', 'infrequent_if_exist', 'error'):
            raise ValueError('Unsupported unseen-level behavior.')
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.encodings_ = {}
        self.output_columns_ = []
        self.removed_columns_ = []
        self.summary_ = []
        used = set(X.columns)
        for column in dict.fromkeys(self.columns):
            series = X[column]
            levels = list(pd.unique(series.dropna()))
            info = {
                'Variable': str(column), 
                'Cardinality':len(levels),
                'Levels':', '.join(map(str, levels)), 
                'Missing':int(series.isna().sum()),
                'Reference':'', 
                'Pooled levels':'', 
                'Indicators':0, 
                'New variables':'', 
                'Status':''
            }
            if not levels:
                info['Status'] = 'No observed levels; original retained.'
                self.summary_.append(info)
                continue
            mapping = {value:i for i,value in enumerate(levels)}
            codes = np.array([mapping[value] for value in series.dropna()], dtype=int).reshape(-1,1)
            encoder = OneHotEncoder(
                drop='if_binary', 
                sparse_output=False, 
                dtype=np.float64,
                handle_unknown=self.handle_unknown, 
                min_frequency=self.min_frequency,
                max_categories=self.max_categories
            ).fit(codes)
            features = list(encoder.get_feature_names_out(['level']))
            pooled = getattr(encoder, 'infrequent_categories_', [None])[0]
            if pooled is not None:
                info['Pooled levels'] = ', '.join(str(levels[int(i)]) for i in pooled)
            # sklearn retains a constant indicator; it carries no level contrast.
            constant = len(features) == 1 and encoder.drop_idx_[0] is None
            if constant:
                features = []
            elif encoder.drop_idx_[0] is not None:
                info['Reference'] = str(levels[int(encoder.drop_idx_[0])])
            names = []
            for feature in features:
                token = feature.removeprefix('level_')
                label = 'infrequent' if token == 'infrequent_sklearn' else str(levels[int(token)])
                base = f'{column}__{label}'
                name, suffix = base, 2
                while name in used:
                    name = f'{base}_{suffix}'
                    suffix += 1
                names.append(name)
                used.add(name)
            if len(self.output_columns_) + len(names) > 4096:
                raise ValueError('Encoding exceeds 4096 indicators. Pool rare levels or reduce the maximum categories.')
            self.encodings_[column] = (mapping, encoder, names)
            self.output_columns_.extend(names)
            if self.remove_original:
                self.removed_columns_.append(column)
            info.update({'Indicators':len(names), 'New variables':', '.join(names),
                'Status':'Constant after grouping; no indicators.' if constant else 'Binary: one reference indicator dropped.' if encoder.drop_idx_[0] is not None else 'One indicator per level.'})
            self.summary_.append(info)
        self.output_features_ = [c for c in X if c not in self.removed_columns_] + self.output_columns_
        return self

    def transform(self, X):
        check_is_fitted(self, 'encodings_')
        self._validate_frame(X)
        if set(self.output_columns_) & set(X.columns):
            raise ValueError('An encoded output name already exists in the incoming data.')
        parts = [X.drop(columns=self.removed_columns_).copy()]
        for column, (mapping, encoder, names) in self.encodings_.items():
            series = X[column]
            missing = series.isna().to_numpy()
            # A known placeholder avoids treating missingness as an unseen level.
            codes = np.array([0 if absent else mapping.get(value,-1)
                for value,absent in zip(series,missing)],dtype=int).reshape(-1,1)
            values = encoder.transform(codes) if len(X) else np.empty((0,len(encoder.get_feature_names_out())))
            if names:
                values[missing,:] = np.nan
                parts.append(pd.DataFrame(values,index=X.index,columns=names))
        return pd.concat(parts,axis=1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'output_features_')
        return np.asarray(self.output_features_,dtype=object)
