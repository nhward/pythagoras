"""Fold-fitted original-variable selection; transform only removes columns."""
import numpy as np
import pandas as pd
from code_recording import recordable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@recordable
class FeatureSelectionTransformer(TransformerMixin, BaseEstimator):
    # Topology/decision policy stays fixed; optional tuning is numerical only.
    tuning_metadata = {'method': {'fixed': True}, 'policy': {'fixed': True},
                       'task': {'fixed': True}, 'structure': {'fixed': True}, 'resampling': {'fixed': True}}
    selection_step = True

    def __init__(self, columns, *, method='Boruta', task='regression', policy='confirmed',
                 trees=200, depth=5, iterations=100, alpha=.05, percentile=100,
                 two_step=True, folds=5, repeats=5, cutoff=1., random_state=2025,
                 n_jobs=1, weighting=(), structure=(), predictors=(), resampling=None):
        self.columns = columns
        self.predictors = predictors
        self.method = method
        self.task = task
        self.policy = policy
        self.trees = trees
        self.depth = depth
        self.iterations = iterations
        self.alpha = alpha
        self.percentile = percentile
        self.two_step = two_step
        self.folds = folds
        self.repeats = repeats
        self.cutoff = cutoff
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.weighting = weighting
        self.structure = structure
        self.resampling = resampling

    def fit(self, X, y=None, *, progress=None):
        from feature_selection import fit_boruta, fit_groot
        from weighting import effective_weights
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Feature selection requires a DataFrame with unique columns')
        if self.method not in ('Boruta', 'GrootCV') or self.task not in ('classification', 'regression'):
            raise ValueError('Selection requires Boruta/GrootCV and a classification or regression task')
        if self.policy not in ('confirmed', 'tentative'):
            raise ValueError('Unknown Boruta inclusion policy')
        for name, minimum in [('trees', 2), ('depth', 1), ('iterations', 3), ('folds', 2), ('repeats', 1), ('n_jobs', 1)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.n_jobs > 32:
            raise ValueError('Use at most 32 selection worker threads')
        if not 0 < self.alpha < 1 or not 0 < self.percentile <= 100 or not np.isfinite(self.cutoff) or self.cutoff <= 0:
            raise ValueError('Invalid significance, shadow percentile or cutoff')
        if y is None:
            raise ValueError('Feature selection needs a target y during fit')
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError('Selection needs one target aligned positionally with X')
        if self.structure and self.method == 'GrootCV' and self.resampling is None:
            raise ValueError('GrootCV random inner folds are unsuitable for declared group/time structure; use Boruta')
        columns = [c for c in X if c in self.columns and c not in self.weighting]
        if not columns:
            raise ValueError('No eligible Predictor columns survive upstream steps')
        labelled = ~pd.isna(y)
        if self.task == 'regression':
            y = np.asarray(y, dtype=float)
            labelled &= np.isfinite(y)
        self.fit_positions_ = np.flatnonzero(labelled)
        if len(self.fit_positions_) < 6:
            raise ValueError('At least six observations with observed outcomes are needed')
        data = X.iloc[self.fit_positions_]
        target = y[self.fit_positions_]
        weight = effective_weights(data, self.weighting) if self.weighting else None
        weights = None if weight is None else weight.to_numpy()
        if weights is not None:
            positive = weights > 0
            self.fit_positions_ = self.fit_positions_[positive]
            data, target, weights = data.iloc[np.flatnonzero(positive)], target[positive], weights[positive]
            weights = weights / weights.mean()
        if self.task == 'classification':
            target, self.classes_ = pd.factorize(target, sort=False)
            counts = np.bincount(target)
            if len(counts) < 2:
                raise ValueError('Classification selection needs at least two outcome classes')
            if self.method == 'GrootCV' and self.resampling is None and counts.min() < self.folds:
                raise ValueError('Each outcome class needs at least the requested inner fold count')
        elif len(np.unique(target)) < 2:
            raise ValueError('Regression selection needs a varying outcome')
        if self.method == 'GrootCV' and self.resampling is None and len(target) < self.folds * 2:
            raise ValueError('Too few labelled observations for the requested inner folds')
        import sys
        # Browser fits and exported recipes must never request native threads.
        self.effective_n_jobs_ = 1 if sys.platform == 'emscripten' else self.n_jobs
        common = dict(weights=weights, task=self.task, trees=self.trees, depth=self.depth,
                      seed=self.random_state, jobs=self.effective_n_jobs_, progress=progress)
        if self.method == 'Boruta':
            details = fit_boruta(data[columns], target, iterations=self.iterations,
                alpha=self.alpha, percentile=self.percentile, two_step=self.two_step, **common)
        else:
            splits = None
            self.inner_resampling_ = {'strategy':'stratified' if self.task == 'classification' else 'kfold',
                                      'origin':'var_select settings', 'notes':[]}
            if self.resampling is not None:
                from selection_resampling import selection_inner_splits
                splits, self.inner_resampling_ = selection_inner_splits(data, target,
                    task=self.task, design=self.resampling, folds=self.folds,
                    repeats=self.repeats, seed=self.random_state)
            details = fit_groot(data[columns], target, folds=self.folds,
                repeats=self.repeats, cutoff=self.cutoff, splits=splits, **common)
            self.inner_splits_ = [(self.fit_positions_[a], self.fit_positions_[b]) for a,b in details['splits']]
        decisions = details['decisions']
        self.support_ = (decisions == 1) | ((decisions == 0) & (self.policy == 'tentative'))
        self.candidate_columns_ = columns
        self.selected_columns_ = [c for c, keep in zip(columns, self.support_) if keep]
        self.removed_columns_ = [c for c, keep in zip(columns, self.support_) if not keep]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.details_ = details
        self.diagnostics_ = pd.DataFrame({'Variable': columns,
            'Decision': [({1:'Confirmed', 0:'Tentative', -1:'Rejected'} if self.method == 'Boruta'
                          else {1:'Selected', -1:'Rejected'})[int(v)] for v in decisions],
            'Included': self.support_, 'Rank': details['ranking'],
            'Mean importance': np.nanmean(details['real'], axis=0),
            'Median importance': np.nanmedian(details['real'], axis=0),
            'Missing proportion': data[columns].isna().mean().to_numpy()})
        if details['hits'] is not None:
            self.diagnostics_['Shadow wins'] = details['hits']
        self.diagnostics_['Fits'] = np.isfinite(details['real']).sum(axis=0)
        self.diagnostics_['Threshold'] = np.mean(details['thresholds'])
        return self

    def transform(self, X):
        check_is_fitted(self, 'selected_columns_')
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Feature selection requires a DataFrame')
        missing = set(self.candidate_columns_) - set(X.columns)
        if missing:
            raise ValueError(f'Missing fitted selection columns: {sorted(missing)}')
        return X.drop(columns=self.removed_columns_).copy()

    def get_support(self, indices=False):
        """Mask (or indices) over fitted candidate columns, excluding passthrough."""
        check_is_fitted(self, 'support_')
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        """Full-frame output names, including protected passthrough columns."""
        check_is_fitted(self, 'selected_columns_')
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        return np.asarray([c for c in names if c not in self.removed_columns_], dtype=object)
