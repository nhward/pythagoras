"""Cross-fitted target encoding of Code predictors, preserving DataFrames."""
from __future__ import annotations

from inspect import signature

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted

METHODS = {
    'auto': 'Empirical Bayes (automatic smoothing)',
    'smooth': 'Smoothed target mean (fixed strength)',
    'mean': 'Target mean (no smoothing)',
}


class TargetEncodingTransformer(TransformerMixin, BaseEstimator):
    """Learn only from fit-time outcomes; transform never reads the target.

    fit_transform uses sklearn's internal cross-fitting for training features.
    fit followed by transform intentionally uses the full fitted mappings and
    is for inference, not for generating features for a training estimator.
    An explicit y overrides the configured target column when fitting.
    """

    def __init__(self, columns, *, target, target_type='continuous', method='auto',
                 smoothing=10., cv=5, remove_original=True, random_state=2025):
        self.columns = columns
        self.target = target
        self.target_type = target_type
        self.method = method
        self.smoothing = smoothing
        self.cv = cv
        self.remove_original = remove_original
        self.random_state = random_state

    def _features(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Target encoding requires a DataFrame with unique column names.')
        if not self.columns or self.target in self.columns:
            raise ValueError('Select Code predictors excluding the Target itself.')
        absent = [c for c in self.columns if c not in X]
        if absent:
            raise ValueError(f'Missing Code predictor columns: {absent}')
        frame = X[list(self.columns)].astype(object)
        return frame.where(frame.notna(), np.nan)

    def _outcomes(self, X, y):
        if y is None:
            if self.target not in X:
                raise ValueError(f'Target {self.target!r} is required during fitting, or supply y.')
            y = X[self.target]
        values = np.asarray(y)
        if values.ndim != 1 or len(values) != len(X):
            raise ValueError('The Target must be one-dimensional and aligned with the fitting rows.')
        values = pd.Series(values)
        if values.isna().any():
            raise ValueError('The Target contains missing outcomes. Resolve them upstream before target encoding.')
        if len(values) < 2:
            raise ValueError('Target encoding needs at least two fitting observations.')
        if self.target_type == 'continuous':
            outcomes = pd.to_numeric(values, errors='raise').to_numpy(dtype=float)
            if not np.isfinite(outcomes).all():
                raise ValueError('The numeric Target must contain finite outcomes.')
            self.classes_ = None
            self.encoder_target_type_ = 'continuous'
            available_folds = len(values)
        elif self.target_type == 'classification':
            # Factorize only observed outcomes, never unused categorical metadata.
            outcomes, classes = pd.factorize(values, sort=False)
            if len(classes) < 2:
                raise ValueError('A categorical Target needs at least two observed classes.')
            self.classes_ = np.asarray(classes, dtype=object)
            self.encoder_target_type_ = 'binary' if len(classes) == 2 else 'multiclass'
            available_folds = int(np.bincount(outcomes).min())
            if available_folds < 2:
                raise ValueError('Cross-fitting needs at least two observations in every Target class.')
        else:
            raise ValueError('Target type must be continuous or classification.')
        self.cv_ = min(int(self.cv), available_folds)
        if self.cv_ < 2:
            raise ValueError('Cross-fitting requires at least two folds.')
        return outcomes

    def _fit(self, X, y, cross_fit):
        features = self._features(X)
        if self.method not in METHODS:
            raise ValueError('Unknown target-encoding method.')
        if not np.isfinite(self.smoothing) or self.smoothing < 0:
            raise ValueError('Smoothing strength must be nonnegative and finite.')
        outcomes = self._outcomes(X, y)
        width = len(self.classes_) if self.encoder_target_type_ == 'multiclass' else 1
        if len(self.columns) * width > 4096:
            raise ValueError('Target encoding exceeds 4096 output features; reduce the number of classes or predictors.')
        smooth = 'auto' if self.method == 'auto' else float(self.smoothing) if self.method == 'smooth' else 0.
        # sklearn 1.9 moved shuffle/seed into a CV splitter. Keep support for
        # the project's >=1.4 baseline, including older Shinylive runtimes.
        shuffle_parameter = signature(TargetEncoder).parameters.get('shuffle')
        if shuffle_parameter is None or shuffle_parameter.default == 'deprecated':
            splitter = KFold if self.encoder_target_type_ == 'continuous' else StratifiedKFold
            cv_options = {'cv':splitter(self.cv_,shuffle=True,random_state=self.random_state)}
        else:
            cv_options = {'cv':self.cv_, 'shuffle':True, 'random_state':self.random_state}
        self.encoder_ = TargetEncoder(target_type=self.encoder_target_type_, smooth=smooth, **cv_options)
        values = self.encoder_.fit_transform(features, outcomes) if cross_fit else None
        if not cross_fit:
            self.encoder_.fit(features, outcomes)
        used = set(X.columns)
        self.output_columns_ = []
        self.summary_ = []
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.removed_columns_ = list(self.columns) if self.remove_original else []
        for column in self.columns:
            names = []
            for class_index in range(width):
                label = '' if self.classes_ is None else str(self.classes_[class_index if width > 1 else 1])
                base = f'{column}__target' + (f'__{label}' if self.classes_ is not None else '')
                name, suffix = base, 2
                while name in used:
                    name = f'{base}_{suffix}'
                    suffix += 1
                used.add(name)
                names.append(name)
            self.output_columns_.extend(names)
            self.summary_.append({'Variable':str(column), 'Cardinality':int(X[column].nunique(dropna=True)),
                'Levels':', '.join(map(str,pd.unique(X[column].dropna()))),
                'Missing':int(X[column].isna().sum()), 'Target':str(self.target),
                'Method':METHODS[self.method], 'Folds':self.cv_, 'Outputs':len(names),
                'New variables':', '.join(names),
                'Meaning':'Target mean' if self.classes_ is None else 'Class probabilities: '+', '.join(map(str,self.classes_ if width > 1 else self.classes_[1:]))})
        self.output_features_ = [c for c in X if c not in self.removed_columns_] + self.output_columns_
        return values

    def fit(self, X, y=None):
        self._fit(X,y,False)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if fit_params:
            raise TypeError('Additional fit parameters are not supported.')
        values = self._fit(X,y,True)
        return self._append(X,values)

    def _append(self, X, values):
        if set(X.columns) & set(self.output_columns_):
            raise ValueError('A target-encoded output name already exists in the incoming data.')
        return pd.concat([X.drop(columns=self.removed_columns_).copy(),
            pd.DataFrame(values,index=X.index,columns=self.output_columns_)],axis=1)

    def transform(self, X):
        check_is_fitted(self,'encoder_')
        features = self._features(X)
        values = self.encoder_.transform(features) if len(X) else np.empty((0,len(self.output_columns_)))
        return self._append(X,values)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self,'output_features_')
        return np.asarray(self.output_features_,dtype=object)
