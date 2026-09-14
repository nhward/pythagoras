"""Train-only clustering followed by a nominal membership feature for new rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted


class ClusterMembershipTransformer(TransformerMixin, BaseEstimator):
    """Append an unordered categorical feature without refitting in transform.

    ``columns`` names eligible predictors, not targets or observation weights.
    Fitting may sample complete positive-weight rows; transform assigns every
    complete row using the fitted preprocessing and model. The weighting column
    is only needed during fit and is not a clustering coordinate.
    """

    def __init__(self, columns, *, n_clusters=2, method="Partition", centre="centroids",
                 metric="euclidean", standardize=True, weighting=None, limit=1000,
                 output_column="cluster", random_state=2025, min_points=5, border=True):
        self.min_points = min_points
        self.border = border
        self.columns = columns
        self.n_clusters = n_clusters
        self.method = method
        self.centre = centre
        self.metric = metric
        self.standardize = standardize
        self.weighting = weighting
        self.limit = limit
        self.output_column = output_column
        self.random_state = random_state

    def _frame(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClusterMembershipTransformer requires a pandas DataFrame")
        if not X.columns.is_unique:
            raise ValueError("Clustering requires unique column names")
        if self.output_column in X.columns:
            raise ValueError(f"Output column {self.output_column!r} already exists")

    def fit(self, X, y=None):
        from cards.obs_k_clusters import _importance_weights, _pam

        self._frame(X)
        if self.method not in ("Partition", "Mixture", "Density"):
            raise ValueError("Only Partition, Mixture and Density support learned membership")
        if self.centre not in ("centroids", "medoids") or self.metric not in ("euclidean", "manhattan"):
            raise ValueError("Unsupported partition center or distance metric")
        if isinstance(self.n_clusters, bool) or not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        if not isinstance(self.limit, (int, np.integer)) or self.limit < 1:
            raise ValueError("limit must be a positive integer")
        if not isinstance(self.output_column, str) or not self.output_column.strip():
            raise ValueError("output_column must be a nonempty string")
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing predictor columns: {sorted(missing)}")
        columns = [c for c in self.columns if c != self.weighting and not str(c).startswith("shadow__")
                   and pd.api.types.is_numeric_dtype(X[c].dtype)
                   and not pd.api.types.is_bool_dtype(X[c].dtype)
                   and not pd.api.types.is_complex_dtype(X[c].dtype)]
        raw_weights = None
        if self.weighting is not None:
            if self.weighting not in X:
                raise ValueError("The fitting data lacks the assigned weighting column")
            w = X[self.weighting]
            if (not pd.api.types.is_numeric_dtype(w.dtype) or pd.api.types.is_bool_dtype(w.dtype)
                    or pd.api.types.is_complex_dtype(w.dtype)):
                raise ValueError("Observation importance must be numeric")
            raw_weights = w.to_numpy(dtype=float, na_value=np.nan)
            _importance_weights(raw_weights)
        positive = np.ones(len(X), dtype=bool) if raw_weights is None else raw_weights > 0
        frame = X[columns].astype(float).replace([np.inf, -np.inf], np.nan)
        frame = frame.loc[:, frame.iloc[np.flatnonzero(positive)].nunique() > 1]
        positions = np.flatnonzero(positive & frame.notna().all(axis=1).to_numpy())
        if len(positions) > self.limit:
            positions = np.sort(np.random.default_rng(self.random_state).choice(positions, self.limit, replace=False))
        x = frame.iloc[positions].to_numpy()
        if not len(x) or not x.shape[1]:
            raise ValueError("Clustering requires complete rows and varying numeric predictors")
        weights = None if raw_weights is None else _importance_weights(raw_weights[positions])
        if weights is not None and np.all(weights == weights[0]):
            weights = None
        if self.method == "Mixture" and weights is not None:
            raise ValueError("Mixture does not support unequal observation importance; disable weighting")
        scale = np.max(np.abs(x), axis=0)
        safe = x / np.where(scale > 0, scale, 1)
        mean = np.average(safe, axis=0, weights=weights)
        spread = np.sqrt(np.average((safe - mean)**2, axis=0, weights=weights))
        keep = spread > 0
        self.predictors_ = list(frame.columns[keep])
        self.scale_ = scale[keep]
        self.mean_ = mean[keep]
        self.spread_ = spread[keep]
        x = (safe[:, keep] - self.mean_) / self.spread_ if self.standardize else x[:, keep]
        if not x.shape[1] or len(np.unique(x, axis=0)) < self.n_clusters:
            raise ValueError("Fewer distinct fitting rows than the requested number of clusters")
        if isinstance(self.min_points, bool) or not isinstance(self.min_points, (int, np.integer)) or self.min_points < 1:
            raise ValueError("min_points must be a positive integer")
        self.model_ = None
        self.medoids_ = None
        if self.method == "Density":
            from cards.obs_clusters import _density
            distance = squareform(pdist(x, metric="cityblock" if self.metric == "manhattan" else "euclidean"))
            if not np.isfinite(distance).all():
                raise ValueError("Distances overflowed; enable standardization")
            _, self.density_note_, model = _density(distance, self.n_clusters, self.min_points, weights, self.border, return_model=True)
            self.radius_ = model.eps
            self.core_points_ = x[model.core_sample_indices_].copy()
            self.core_labels_ = model.labels_[model.core_sample_indices_].copy()
            self.density_points_ = x.copy()
            self.density_weights_ = np.ones(len(x)) if weights is None else weights.copy()
            labels = model.labels_.copy()
        elif self.n_clusters == 1:
            labels = np.zeros(len(x), dtype=int)
        elif self.method == "Mixture":
            candidates = []
            for covariance in ("spherical", "diag", "tied", "full"):
                try:
                    candidate = GaussianMixture(n_components=self.n_clusters, covariance_type=covariance,
                        random_state=self.random_state, n_init=2, max_iter=300).fit(x)
                    if candidate.converged_ and np.isfinite(candidate.bic(x)):
                        candidates.append(candidate)
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if not candidates:
                raise ValueError("No Gaussian mixture converged")
            self.model_ = min(candidates, key=lambda model: model.bic(x))
            labels = self.model_.predict(x)
        elif self.centre == "medoids":
            distance = squareform(pdist(x, metric="cityblock" if self.metric == "manhattan" else "euclidean"))
            labels, medoids = _pam(distance, self.n_clusters, weights, return_medoids=True)
            self.medoids_ = x[medoids].copy()
        else:
            self.model_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(x, sample_weight=weights)
            labels = self.model_.labels_
        if self.method != "Density" and len(np.unique(labels)) != self.n_clusters:
            raise ValueError("The fitted model did not produce the requested occupied groups")
        self.label_map_ = {int(label): f"c{i + 1}" for i, label in enumerate(dict.fromkeys(labels[labels >= 0]))}
        if self.method == "Density":
            self.label_map_[-1] = "unallocated"
        self.fit_positions_ = positions
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "label_map_")
        self._frame(X)
        missing = set(self.predictors_) - set(X.columns)
        if missing:
            raise ValueError(f"Missing fitted predictor columns: {sorted(missing)}")
        raw = X[self.predictors_].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(raw).all(axis=1)
        values = np.full(len(X), None, dtype=object)
        if valid.any():
            x = raw[valid]
            if self.standardize:
                x = (x / self.scale_ - self.mean_) / self.spread_
            if not np.isfinite(x).all():
                raise ValueError("Transformed predictor values overflowed")
            if self.method == "Density":
                labels = np.full(len(x), -1, dtype=int)
                metric = "cityblock" if self.metric == "manhattan" else "euclidean"
                # Bound temporary distance memory for large prediction batches.
                for start in range(0, len(x), 512):
                    batch = x[start:start + 512]
                    if not len(self.core_points_):
                        continue
                    distances = cdist(batch, self.core_points_, metric=metric)
                    nearest = distances.argmin(axis=1)
                    allocated = distances[np.arange(len(batch)), nearest] <= self.radius_
                    if not self.border:
                        training_distances = cdist(batch, self.density_points_, metric=metric)
                        mass = (training_distances <= self.radius_) @ self.density_weights_
                        # Assess density against the fixed training reference only.
                        allocated &= mass >= self.min_points
                    labels[start:start + len(batch)][allocated] = self.core_labels_[nearest[allocated]]
            elif self.n_clusters == 1:
                labels = np.zeros(len(x), dtype=int)
            elif self.medoids_ is not None:
                labels = cdist(x, self.medoids_, metric="cityblock" if self.metric == "manhattan" else "euclidean").argmin(axis=1)
            else:
                labels = self.model_.predict(x)
            values[valid] = [self.label_map_[int(label)] for label in labels]
        result = X.copy()
        result[self.output_column] = pd.Categorical(values, categories=list(self.label_map_.values()), ordered=False)
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features must match the fitting columns")
        return np.append(self.feature_names_in_, self.output_column)
