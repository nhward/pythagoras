"""Ordered ranks and equal-score orthogonal polynomial contrasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

METHODS = {'ordinal':'Ordinal ranks', 'polynomial':'Orthogonal polynomial contrasts'}


def polynomial_contrasts(levels, degree=None):
    """Orthonormal contrasts on equally spaced ranks (R contr.poly convention).

    Multiplication by rank followed by reorthogonalization builds increasing
    polynomial degrees without an ill-conditioned monomial Vandermonde matrix.
    Positive normalization gives each polynomial a positive leading coefficient.
    """
    degree = levels-1 if degree is None else min(int(degree), levels-1)
    if levels < 2 or degree < 1:
        return np.empty((levels,0))
    x = np.linspace(-1.,1.,levels)
    basis = np.empty((levels,degree+1))
    basis[:,0] = 1 / np.sqrt(levels)
    for k in range(1,degree+1):
        values = x * basis[:,k-1]
        previous = basis[:,:k]
        for _ in range(2):
            values -= previous @ (previous.T @ values)
        basis[:,k] = values / np.linalg.norm(values)
    return basis[:,1:]


class OrderedEncodingTransformer(TransformerMixin, BaseEstimator):
    """Preserve the declared order, including levels absent in a training fold.

    Category order is schema supplied upstream, not inferred from outcomes or
    frequencies. OrdinalEncoder receives rank tokens to support explicitly
    ordered numeric labels that are not sorted by numerical value.
    """

    def __init__(self, columns, *, method='ordinal', degree=0,
                 remove_original=True, handle_unknown='missing'):
        self.columns = columns
        self.method = method
        self.degree = degree
        self.remove_original = remove_original
        self.handle_unknown = handle_unknown

    def _validate(self,X):
        if not isinstance(X,pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Ordered encoding requires a DataFrame with unique column names.')
        absent = [c for c in self.columns if c not in X]
        if absent:
            raise ValueError(f'Missing Ordered predictor columns: {absent}')

    @staticmethod
    def _tokens(series, levels):
        missing = series.isna().to_numpy()
        codes = pd.Index(levels).get_indexer(series).astype(float)
        codes[missing] = 0  # Mask missing rows separately, even if fit saw none.
        return codes.reshape(-1,1),missing

    def fit(self,X,y=None):
        self._validate(X)
        if self.method not in METHODS or self.handle_unknown not in ('missing','error'):
            raise ValueError('Unsupported Ordered encoding method or unseen-level behavior.')
        if not isinstance(self.degree,(int,np.integer)) or self.degree < 0:
            raise ValueError('Polynomial degree must be a nonnegative integer; zero means all contrasts.')
        if not len(X):
            raise ValueError('Ordered encoding needs at least one observation.')
        self.encodings_ = {}
        self.output_columns_ = []
        self.removed_columns_ = []
        self.summary_ = []
        self.feature_names_in_ = np.asarray(X.columns,dtype=object)
        self.n_features_in_ = len(X.columns)
        used = set(X.columns)
        for column in dict.fromkeys(self.columns):
            series = X[column]
            if not isinstance(series.dtype,pd.CategoricalDtype) or not series.cat.ordered:
                raise ValueError(f'{column!r} must have an explicitly ordered categorical dtype during fitting.')
            levels = list(series.cat.categories)
            d = len(levels)
            row = {
                    'Variable':str(column),
                    'Cardinality':d,
                    'Observed levels':int(series.nunique()),
                    'Levels (low → high)':' → '.join(map(str,levels)), 
                    'Missing':int(series.isna().sum()),
                    'Method':METHODS[self.method], 
                    'Outputs':0, 
                    'New variables':'', 
                    'Status':''
                }
            if d < 2:
                row['Status'] = 'Fewer than two declared levels; original retained.'
                self.summary_.append(row)
                continue
            degree = min(self.degree or d-1,d-1)
            if self.method == 'polynomial' and degree > 64:
                raise ValueError('Polynomial encoding supports at most 64 degrees. Limit the degree or use ordinal ranks.')
            codes,_ = self._tokens(series,levels)
            encoder = OrdinalEncoder(
                categories=[np.arange(d)],
                dtype=float,
                handle_unknown='error' if self.handle_unknown == 'error' else 'use_encoded_value',
                unknown_value=None if self.handle_unknown == 'error' else np.nan
            ).fit(codes)
            contrasts = polynomial_contrasts(d,degree) if self.method == 'polynomial' else None
            suffixes = ['rank'] if contrasts is None else ['L' if k==1 else 'Q' if k==2 else 'C' if k==3 else f'P{k}' for k in range(1,degree+1)]
            names = []
            for suffix in suffixes:
                base = f'{column}__{suffix}'
                name,number = base,2
                while name in used:
                    name = f'{base}_{number}'
                    number += 1
                used.add(name)
                names.append(name)
            if len(self.output_columns_) + len(names) > 4096:
                raise ValueError('Ordered encoding exceeds 4096 features. Reduce polynomial degree.')
            self.encodings_[column] = (levels,encoder,contrasts,names)
            self.output_columns_.extend(names)
            if self.remove_original:
                self.removed_columns_.append(column)
            row.update({'Outputs':len(names),'New variables':', '.join(names),
                'Status':'Ranks 0 to d−1 in declared order.' if contrasts is None else 'Equal-spacing polynomial contrasts; no constant column.'})
            self.summary_.append(row)
        self.output_features_ = [c for c in X if c not in self.removed_columns_] + self.output_columns_
        return self

    def transform(self,X):
        check_is_fitted(self,'encodings_')
        self._validate(X)
        if set(self.output_columns_) & set(X.columns):
            raise ValueError('An Ordered output name already exists in the incoming data.')
        parts = [X.drop(columns=self.removed_columns_).copy()]
        for column,(levels,encoder,contrasts,names) in self.encodings_.items():
            codes,missing = self._tokens(X[column],levels)
            ranks = encoder.transform(codes).ravel() if len(X) else np.array([])
            ranks[missing] = np.nan
            values = ranks[:,None]
            if contrasts is not None:
                values = np.full((len(X),len(names)),np.nan)
                valid = np.isfinite(ranks)
                values[valid,:] = contrasts[ranks[valid].astype(int)]
            parts.append(pd.DataFrame(values,index=X.index,columns=names))
        return pd.concat(parts,axis=1)

    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,'output_features_')
        return np.asarray(self.output_features_,dtype=object)
