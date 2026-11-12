"""Coordinate selected cluster memberships under one shared cluster-count knob."""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from bundled_parameters import BundledParametersMixin
from ClusterMembershipTransformer import ClusterMembershipTransformer
from code_recording import recordable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@recordable
class ClusterMembershipsTransformer(BundledParametersMixin, TransformerMixin, BaseEstimator):
    """Fit each selected method on the same input; append all memberships.

    Method-specific configuration is nested under its owning branch. Fitted
    children are not constructor parameters: sklearn discovers one shared
    n_clusters, never a separate K for each method.
    """
    tuning_metadata: ClassVar[dict[str, dict[str, object]]] = {
        'n_clusters': {'kind': 'integer', 'label': 'K (shared cluster count)',
                      'suggested': {'values': [2, 3, 5, 8]}, 'importance': 100,
                      'reason': 'Shared K for Partition and Mixture; target group count for Density.'},
        'methods': {'fixed': True}, 'output_columns': {'fixed': True},
        'centre': {'fixed': True}, 'metric': {'fixed': True},
        'standardize': {'fixed': True}, 'weighting': {'fixed': True},
        'density_min_points': {'fixed': True, 'kind': 'integer',
            'reason': 'Density minimum-points modeling setting, held fixed in this version; available for future tuning.'},
        'density_border': {'fixed': True, 'kind': 'boolean',
            'reason': 'Density border-allocation modeling setting, held fixed in this version; available for future tuning.'},
    }

    parameter_groups: ClassVar[dict[str, dict[str, object]]] = {'partition': {'selector': 'method',
                   'values': ['Partition'],
                   'parameters': {'centre': 'center'}},
     'distance': {'selector': 'method',
                  'values': ['Partition', 'Density'],
                  'parameters': {'metric': 'metric'}},
     'density': {'selector': 'method',
                 'values': ['Density'],
                 'parameters': {'min_points': 'min_points', 'border': 'border'}}}

    def __init__(self, columns, *, n_clusters=2, methods=('Partition',),
                 output_columns=None, centre='centroids', metric='euclidean',
                 standardize=True, weighting=None, random_state=2025,
                 density_min_points=5, density_border=True, partition=None, distance=None, density=None):
        self.columns = columns
        self.n_clusters = n_clusters
        self.methods = methods
        self.output_columns = output_columns
        self.centre = centre
        self.metric = metric
        self.standardize = standardize
        self.weighting = weighting
        self.random_state = random_state
        self.density_min_points = density_min_points
        self.density_border = density_border
        self._init_parameter_groups(partition=partition, distance=distance, density=density)


    def _outputs(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Memberships require a DataFrame with unique columns.')
        if isinstance(self.methods, str) or not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError('Choose a nonempty, unique sequence of membership methods.')
        if set(self.methods) - {'Partition', 'Mixture', 'Density'}:
            raise ValueError('Only Partition, Mixture and Density support learned membership.')
        names = dict(self.output_columns) if self.output_columns is not None else {
            method: f'cluster__{method.lower()}' for method in self.methods}
        if set(names) != set(self.methods) or len(set(names.values())) != len(names):
            raise ValueError('Provide one distinct output column for each selected method.')
        if any(not isinstance(name, str) or not name.strip() or name.startswith('shadow__') for name in names.values()):
            raise ValueError('Membership names must be nonempty and cannot use the shadow prefix.')
        if set(names.values()) & set(X.columns):
            raise ValueError('Membership output columns already exist in the input.')
        return names

    def fit(self, X, y=None):
        names = self._outputs(X)
        # Publish fitted state only after every selected method succeeds.
        models = {}
        for method in self.methods:
            models[method] = ClusterMembershipTransformer(
                self.columns, n_clusters=self.n_clusters, method=method,
                output_column=names[method], centre=self.parameter_value('centre'), metric=self.parameter_value('metric'),
                standardize=self.standardize, weighting=self.weighting,
                random_state=self.random_state, min_points=self.parameter_value('density_min_points'),
                border=self.parameter_value('density_border')).fit(X, y)
        self.models_ = models
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, 'models_')
        self._outputs(X)
        result = X.copy()
        for model in self.models_.values():
            # Each child sees original predictors, not another child's output.
            result[model.output_column] = model.transform(X)[model.output_column]
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'models_')
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError('input_features must match the fitting columns')
        return np.append(self.feature_names_in_, [m.output_column for m in self.models_.values()])
