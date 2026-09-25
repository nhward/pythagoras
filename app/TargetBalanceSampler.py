"""Training-only, role-preserving target balancing for imblearn pipelines.

X is a complete role-bearing DataFrame, not just the model's predictors.
Keep the final role/weight adapter downstream of this sampler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from code_recording import recordable
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from var_types import var_kind


@recordable
def balance_weights(frame, column=None):
    """Importance weights must be finite, nonnegative and have positive mass."""
    values = (np.ones(len(frame)) if column is None else
              pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan))
    if not np.isfinite(values).all() or (values < 0).any() or not np.isfinite(values.sum()) or not values.sum() > 0:
        raise ValueError("Weights must be finite, nonnegative, and have positive total weight.")
    return values


@recordable
class BalanceSpace:
    """Numeric standardized distances plus unit mismatches for nominal values."""
    def __init__(self, frame, columns, normalize, metric):
        self.columns = list(columns)
        self.numeric = [c for c in columns if var_kind(frame[c]) in ("integer", "decimal")]
        self.nominal = [c for c in columns if c not in self.numeric]
        numeric = frame[self.numeric].astype(float)
        self.center = numeric.mean() if normalize else pd.Series(0., index=self.numeric)
        self.scale = numeric.std(ddof=0) if normalize else pd.Series(1., index=self.numeric)
        self.scale = self.scale.where(self.scale > 0, 1.)
        self.levels = {c: list(pd.unique(frame[c])) for c in self.nominal}
        if len(frame) * (len(self.numeric) + sum(map(len, self.levels.values()))) > 4_000_000:
            raise ValueError("Encoded distance data exceeds four million cells; reduce or encode high-cardinality predictors upstream.")
        self.metric = metric

    def transform(self, frame):
        parts = [((frame[self.numeric].astype(float) - self.center) / self.scale).to_numpy()]
        # A categorical mismatch contributes one to squared Euclidean distance
        # or one to Manhattan distance, independent of the number of levels.
        divisor = np.sqrt(2) if self.metric == "euclidean" else 2.
        for c in self.nominal:
            parts.append(np.column_stack([(frame[c] == level).to_numpy(dtype=float)
                                          for level in self.levels[c]]) / divisor)
        return np.column_stack(parts)


@recordable
def balance_clusters(values, count, metric, seed, iterations):
    """K-means for Euclidean; alternating coordinate medians for Manhattan."""
    fitted = KMeans(n_clusters=count, random_state=seed, n_init=5, max_iter=iterations).fit(values)
    centers = fitted.cluster_centers_
    if metric == "manhattan":
        for _ in range(iterations):
            labels = pairwise_distances(values, centers, metric=metric).argmin(axis=1)
            updated = np.array([np.median(values[labels == k], axis=0) if (labels == k).any()
                                else centers[k] for k in range(count)])
            if np.allclose(updated, centers):
                break
            centers = updated
    return centers, pairwise_distances(values, centers, metric=metric).argmin(axis=1)


@recordable
def balance_medoids(values, count, metric, iterations, seed):
    """Bounded alternating k-medoids, retaining distinct original positions."""
    distance = pairwise_distances(values, metric=metric)
    rng = np.random.default_rng(seed)
    medoids = [int(rng.integers(len(values)))]
    nearest = distance[:, medoids[0]].copy()
    while len(medoids) < count:
        nearest[medoids] = -1
        chosen = int(nearest.argmax())
        medoids.append(chosen)
        nearest = np.minimum(nearest, distance[:, chosen])
    medoids = np.array(medoids)
    for _ in range(iterations):
        labels = distance[:, medoids].argmin(axis=1)
        labels[medoids] = np.arange(count)  # Keep identical-point medoids distinct.
        updated = medoids.copy()
        for k in range(count):
            members = np.flatnonzero(labels == k)
            updated[k] = members[distance[np.ix_(members, members)].sum(axis=1).argmin()]
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    return np.sort(medoids)


@recordable
class TargetBalanceSampler(BaseEstimator):
    """Cloneable recipe: fit_resample learns only from its supplied training set.

    count_fraction locates the desired class count between observed min and max
    counts, recomputed independently in each fit. No transform method is exposed:
    imblearn skips this step at inference. evaluation_weights is an explicit,
    separately invoked policy; the sampler does not route model sample_weight.
    """
    def __init__(self, target, predictors=(), weight=None, output_weight="Weights",
                 mode="reweight", up="random", down="random", count_fraction=.5,
                 metric="euclidean", normalize=True, evaluation="none",
                 voting="hard", neighbors=5, medoid_limit=2000,
                 iterations=30, random_state=2025, max_rows=100000,
                 distance_limit=10000):
        self.target = target
        self.predictors = predictors
        self.weight = weight
        self.output_weight = output_weight
        self.mode = mode
        self.up = up
        self.down = down
        self.count_fraction = count_fraction
        self.metric = metric
        self.normalize = normalize
        self.evaluation = evaluation
        self.voting = voting
        self.neighbors = neighbors
        self.medoid_limit = medoid_limit
        self.iterations = iterations
        self.random_state = random_state
        self.max_rows = max_rows
        self.distance_limit = distance_limit

    def fit(self, X, y=None):
        self.fit_resample(X, y)
        return self

    def fit_resample(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("Balancing requires a DataFrame with unique column names.")
        if self.mode not in ("reweight", "resample"):
            raise ValueError("Choose reweight or resample.")
        if self.metric not in ("euclidean", "manhattan") or not 0 <= self.count_fraction <= 1:
            raise ValueError("Invalid distance metric or desired count fraction.")
        if self.up not in ("random", "smote") or self.down not in (
                "random", "medoids", "centroids", "nearmiss", "stratified"):
            raise ValueError("Unknown resampling method.")
        if self.medoid_limit > 5000:
            raise ValueError("The medoid row limit cannot exceed 5,000.")
        if self.voting not in ("hard", "soft") or self.evaluation not in ("none", "incoming", "balanced"):
            raise ValueError("Invalid voting or evaluation policy.")
        if min(self.neighbors, self.medoid_limit, self.iterations, self.max_rows, self.distance_limit) < 1:
            raise ValueError("Resource limits and iteration counts must be positive.")
        if var_kind(X[self.target]) != "nominal":
            raise ValueError("Target balancing requires a nominal Target.")
        if y is not None and (len(y) != len(X) or not np.array_equal(
                pd.Series(y).astype(object).fillna("__missing_target__").to_numpy(),
                X[self.target].astype(object).fillna("__missing_target__").to_numpy())):
            raise ValueError("y must match the Target column positionally.")
        weights = balance_weights(X, self.weight)
        self.warnings_ = []
        self.dropped_ = 0
        self.synthetic_ = 0
        self.ignored_ = []
        valid = X[self.target].notna().to_numpy(copy=True)
        # Missing labels cannot define a training class. Reweight leaves these
        # rows intact with factor one; resampling omits them explicitly.
        observed = X[self.target].cat.remove_unused_categories()
        classes = list(observed.cat.categories)
        if len(classes) < 2:
            raise ValueError("At least two observed target classes are required.")
        totals = np.bincount(observed.cat.codes.to_numpy()[valid], weights=weights[valid], minlength=len(classes))
        if (totals <= 0).any():
            raise ValueError("Every observed target class must have positive total weight.")
        self.class_factors_ = dict(zip(classes, totals.sum() / (len(classes) * totals)))
        self.classes_ = classes
        if self.mode == "reweight":
            result = X.copy()
            if self.weight is None and self.output_weight in X:
                raise ValueError("The new weight column would overwrite an existing variable.")
            factors = np.ones(len(X))
            factors[valid] = (totals.sum() / (len(classes) * totals))[observed.cat.codes.to_numpy()[valid]]
            result[self.weight or self.output_weight] = weights * factors
            self.sample_indices_ = np.arange(len(X))
            if not valid.all():
                self.warnings_.append("Missing targets retain their incoming weight; excluded from the diagnostic.")
            return result, result[self.target].copy()

        if len(X) > self.max_rows:
            raise ValueError(f"Resampling is limited to {self.max_rows:,} input rows; reduce the upstream data.")
        counts = X[self.target].value_counts().reindex(classes).to_numpy()
        initial_desired = int(np.floor(counts.min() + self.count_fraction * (counts.max() - counts.min()) + .5))
        need_up, need_down = (counts < initial_desired).any(), (counts > initial_desired).any()
        distance = (need_up and self.up == "smote") or (need_down and self.down != "random")
        supported = []
        if distance:
            supported = [c for c in self.predictors if c in X and c not in (self.target, self.weight) and var_kind(X[c]) in
                         ("decimal", "integer", "nominal", "logical")]
            self.ignored_ = [c for c in self.predictors if c not in supported]
            if self.ignored_:
                self.warnings_.append("Ignored distance predictors: " + ", ".join(map(str, self.ignored_)) +
                                      ". Encode unsupported types upstream (geometry has no encoder).")
            if not supported:
                raise ValueError("No supported predictors for distance resampling; encode variables or use random methods.")
            for c in supported:
                valid &= X[c].notna().to_numpy()
                if var_kind(X[c]) in ("decimal", "integer"):
                    valid &= np.isfinite(X[c].to_numpy(dtype=float, na_value=np.nan))
            if valid.sum() > self.distance_limit:
                raise ValueError(f"Distance resampling is limited to {self.distance_limit:,} complete rows; use random methods or reduce data.")
        positions = np.flatnonzero(valid)
        frame = X.iloc[positions].copy()
        self.dropped_ = len(X) - len(frame)
        if self.dropped_:
            self.warnings_.append(f"Omitted {self.dropped_:,} rows with missing targets or required predictor values.")
        codes = pd.Index(classes).get_indexer(frame[self.target])
        counts = np.bincount(codes, minlength=len(classes))
        if (counts == 0).any():
            raise ValueError("Required-value removal empties a target class; impute upstream or use random resampling.")
        kept_weights = weights[positions]
        if (np.bincount(codes, weights=kept_weights, minlength=len(classes)) <= 0).any():
            raise ValueError("Required-value removal leaves a class with zero total weight.")
        desired = int(np.floor(counts.min() + self.count_fraction * (counts.max() - counts.min()) + .5))
        self.desired_count_ = desired
        if desired * len(classes) > self.max_rows:
            raise ValueError(f"Balanced output would exceed the {self.max_rows:,}-row limit.")
        down = {k: desired for k, count in enumerate(counts) if count > desired}
        space = BalanceSpace(frame, supported, self.normalize, self.metric) if distance else None
        values = space.transform(frame) if space is not None else np.zeros((len(frame), 1))
        rng = np.random.default_rng(self.random_state)
        if down and self.down in ("random", "nearmiss"):
            if self.down == "random":
                sampler = RandomUnderSampler(sampling_strategy=down, random_state=self.random_state)
            else:
                neighbors = NearestNeighbors(n_neighbors=min(self.neighbors, int(counts.min())), metric=self.metric)
                sampler = NearMiss(sampling_strategy=down, n_neighbors=neighbors, version=1)
            sampler.fit_resample(values, codes)
            selected = sampler.sample_indices_
            result = frame.iloc[selected].copy()
            donors = positions[selected]
        elif down:
            pieces, donor_pieces = [], []
            for k in range(len(classes)):
                indices = np.flatnonzero(codes == k)
                block = values[indices]
                original = frame.iloc[indices]
                if k not in down:
                    chosen = np.arange(len(indices))
                    piece = original.copy()
                elif self.down == "medoids":
                    if len(indices) > self.medoid_limit:
                        raise ValueError(f"K-medoids is limited to {self.medoid_limit:,} rows per class; use another method.")
                    chosen = balance_medoids(block, desired, self.metric, self.iterations, self.random_state)
                    piece = original.iloc[chosen].copy()
                else:
                    clusters = min(desired, max(1, int(np.sqrt(len(indices))))) if self.down == "stratified" else desired
                    if len(block) * clusters > 4_000_000:
                        raise ValueError("Clustering exceeds the 4-million row-center distance budget; use random sampling or fewer desired rows.")
                    centers, labels = balance_clusters(block, clusters, self.metric, self.random_state, self.iterations)
                    if self.down == "stratified":
                        groups = [np.flatnonzero(labels == j) for j in range(clusters)]
                        groups = [g for g in groups if len(g)]
                        # One per populated cluster, then proportional residual
                        # allocation without replacement or overfilling a cluster.
                        capacity = np.array([len(g) - 1 for g in groups])
                        quota = np.ones(len(groups), dtype=int)
                        if capacity.sum():
                            exact = (desired - len(groups)) * capacity / capacity.sum()
                            quota += np.floor(exact).astype(int)
                            order = np.argsort(-(exact - np.floor(exact)), kind="stable")
                            quota[order[:desired - quota.sum()]] += 1
                        chosen = np.concatenate([rng.choice(g, n, replace=False) for g, n in zip(groups, quota)])
                        piece = original.iloc[chosen].copy()
                    else:
                        # Hard voting selects the nearest row to each centroid;
                        # repeated donors are possible with coincident centroids.
                        chosen = pairwise_distances(centers, block, metric=self.metric).argmin(axis=1)
                        piece = original.iloc[chosen].copy()
                        if self.voting == "soft":
                            for j, c in enumerate(space.numeric):
                                piece[c] = centers[:, j] * space.scale[c] + space.center[c]
                            offset = len(space.numeric)
                            for c in space.nominal:
                                levels = space.levels[c]
                                modes = []
                                for j in range(clusters):
                                    members = original.iloc[np.flatnonzero(labels == j)][c]
                                    modes.append(members.mode().iloc[0] if len(members) else piece[c].iloc[j])
                                piece[c] = pd.Series(modes, dtype=frame[c].dtype).array
                                offset += len(levels)
                            self.synthetic_ += len(piece)
                pieces.append(piece)
                donor_pieces.append(positions[indices[chosen]])
            result = pd.concat(pieces)
            donors = np.concatenate(donor_pieces)
        else:
            result, donors = frame.copy(), positions.copy()

        result_codes = pd.Index(classes).get_indexer(result[self.target])
        up = {k: desired for k in range(len(classes)) if (result_codes == k).sum() < desired}
        if up and self.up == "random":
            sampler = RandomOverSampler(sampling_strategy=up, random_state=self.random_state)
            sampler.fit_resample(np.zeros((len(result), 1)), result_codes)
            result = result.iloc[sampler.sample_indices_].copy()
            donors = donors[sampler.sample_indices_]
        elif up:
            minimum = min(int((result_codes == k).sum()) for k in up)
            if minimum < 2:
                raise ValueError("SMOTE needs at least two complete rows in each upsampled class; use RandomOverSampler.")
            n = min(self.neighbors, minimum - 1)
            numeric, nominal = space.numeric, space.nominal
            encoded = result[numeric + nominal].copy()
            for c in numeric:
                encoded[c] = (encoded[c].astype(float) - space.center[c]) / space.scale[c]
            for c in nominal:
                encoded[c] = pd.Index(space.levels[c]).get_indexer(encoded[c])
            common = {"sampling_strategy": up, "random_state": self.random_state}
            nearest = NearestNeighbors(n_neighbors=n + 1, metric=self.metric)
            if numeric and nominal:
                sampler = SMOTENC(categorical_features=list(range(len(numeric), encoded.shape[1])),
                                   k_neighbors=nearest, **common)
            elif numeric:
                sampler = SMOTE(k_neighbors=nearest, **common)
            else:
                sampler = SMOTEN(k_neighbors=n, **common)
            sampled, labels = sampler.fit_resample(encoded, result_codes)
            extra = sampled.iloc[len(result):].copy()
            extra_labels = labels[len(result):]
            for c in numeric:
                extra[c] = extra[c] * space.scale[c] + space.center[c]
            for c in nominal:
                extra[c] = pd.Series([space.levels[c][int(i)] for i in extra[c]], dtype=frame[c].dtype).array
            new_donors = np.empty(len(extra), dtype=int)
            # Metadata donor: closest original complete row of the same class.
            for k in up:
                mask = extra_labels == k
                original = np.flatnonzero(codes == k)
                nearest = NearestNeighbors(n_neighbors=1, metric=self.metric).fit(values[original])
                matches = nearest.kneighbors(space.transform(extra.loc[mask]), return_distance=False).ravel()
                new_donors[mask] = positions[original[matches]]
            synthetic = X.iloc[new_donors].copy()
            for c in numeric + nominal:
                synthetic[c] = extra[c].array
            result = pd.concat([result, synthetic])
            donors = np.r_[donors, new_donors]
            self.synthetic_ += len(extra)
        self.sample_indices_ = np.asarray(donors, dtype=int)
        if self.synthetic_:
            self.warnings_.append(f"{self.synthetic_:,} synthetic rows: non-predictor and unsupported values copied from same-class donors; identifiers may repeat.")
        return result, result[self.target].copy()

    def evaluation_weights(self, X):
        """Explicit holdout policy; no rows are added, removed or synthesized."""
        check_is_fitted(self, "class_factors_")
        if self.evaluation == "none":
            return None
        incoming = balance_weights(X, self.weight)
        if self.evaluation == "incoming":
            return incoming
        if X[self.target].isna().any() or not X[self.target].isin(self.classes_).all():
            raise ValueError("Balanced evaluation requires known, nonmissing target classes.")
        return incoming * np.array([self.class_factors_[c] for c in X[self.target]])
