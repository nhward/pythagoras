from __future__ import annotations

import asyncio
import os
import sys
from functools import lru_cache
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from module import Module
from proxy_data import proxy_data
from roles import Role
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.spatial.distance import pdist, squareform
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

FAMILIES = ("Agglomerative", "Divisive", "Partition", "Mixture", "Topology", "Density", "Spectral", "Stability", "Gap statistic")

def _importance_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Importance weights must be finite, non-negative, and include a positive value")
    scaled = weights / weights.max()
    return scaled / scaled.mean()


def _weighted_metrics(x, distance, labels, weights):
    # Importance-weight extensions: retain actual observation count in CH's
    # degrees-of-freedom factor and equal cluster contributions in DB.
    # Neither formula treats sum(weights) as a count of independent observations.
    groups = np.unique(labels)
    centres, scatter, masses = [], [], []
    within = 0.0
    means = np.empty((len(x), len(groups)))
    own = np.empty(len(x), dtype=int)
    for j, group in enumerate(groups):
        mask = labels == group
        w = weights[mask]
        centre = np.average(x[mask], axis=0, weights=w)
        residual = np.linalg.norm(x[mask] - centre, axis=1)
        centres.append(centre)
        scatter.append(np.average(residual, weights=w))
        masses.append(w.sum())
        within += np.dot(w, residual ** 2)
        means[:, j] = distance[:, mask] @ w / w.sum()
        own[mask] = j
    centres = np.asarray(centres)
    masses = np.asarray(masses)
    a = np.zeros(len(x))
    denominator = masses[own] - weights
    valid = denominator > 0
    a[valid] = means[np.arange(len(x)), own][valid] * masses[own][valid] / denominator[valid]
    means[np.arange(len(x)), own] = np.inf
    b = means.min(axis=1)
    silhouette = np.divide(b - a, np.maximum(a, b), out=np.zeros(len(x)), where=np.maximum(a, b) > 0)
    silhouette[~valid] = 0
    centre = np.average(x, axis=0, weights=weights)
    between = np.sum(masses[:, None] * (centres - centre) ** 2)
    ch = between / within * (len(x) - len(groups)) / (len(groups) - 1) if within else 1.0
    separations = squareform(pdist(centres))
    if np.allclose(scatter, 0) or np.allclose(separations, 0):
        db = 0.0
    else:
        separations[separations == 0] = np.inf
        db = np.mean(((np.asarray(scatter)[:, None] + scatter) / separations).max(axis=1))
    return (("Silhouette", float(np.average(silhouette, weights=weights)), "maximize"),
            ("Calinski–Harabasz", float(ch), "maximize"),
            ("Davies–Bouldin", float(db), "minimize"))


def _diana(distance: np.ndarray, maximum: int) -> dict[int, np.ndarray]:
    # Kaufman & Rousseeuw's DIANA: split the greatest-diameter cluster;
    # move points to the splinter while their mean-distance difference is positive.
    groups = [np.arange(len(distance))]
    results = {1: np.zeros(len(distance), dtype=int)}
    while len(groups) < maximum:
        index = max(
            (i for i, group in enumerate(groups) if len(group) > 1),
            key=lambda i: distance[np.ix_(groups[i], groups[i])].max(),
        )
        group = groups.pop(index)
        sub = distance[np.ix_(group, group)]
        seed = int(np.argmax(sub.sum(axis=1)))
        splinter = [int(group[seed])]
        remaining = list(np.delete(group, seed))
        while len(remaining) > 1:
            within = distance[np.ix_(remaining, remaining)].sum(axis=1) / (len(remaining) - 1)
            between = distance[np.ix_(remaining, splinter)].mean(axis=1)
            gains = within - between
            candidate = int(np.argmax(gains))
            if gains[candidate] <= 0:
                break
            splinter.append(remaining.pop(candidate))
        groups.extend([np.asarray(remaining), np.asarray(splinter)])
        labels = np.empty(len(distance), dtype=int)
        for label, members in enumerate(groups):
            labels[members] = label
        results[len(groups)] = labels
    return results


def _pam(distance: np.ndarray, k: int, weights=None, *, return_medoids=False) -> np.ndarray:
    # https://stat.ethz.ch/CRAN/web/packages/cluster/refman/cluster.html
    # PAM BUILD followed by best-improving SWAP (Kaufman & Rousseeuw).
    weights = np.ones(len(distance)) if weights is None else weights
    medoids = [int(np.argmin(weights @ distance))]
    nearest = distance[:, medoids[0]].copy()
    while len(medoids) < k:
        costs = weights @ np.minimum(nearest[:, None], distance)
        costs[medoids] = np.inf
        medoids.append(int(np.argmin(costs)))
        nearest = np.minimum(nearest, distance[:, medoids[-1]])
    for _ in range(100):
        distances = distance[:, medoids]
        owner = distances.argmin(axis=1)
        nearest = distances.min(axis=1)
        second = np.partition(distances, 1, axis=1)[:, 1] if k > 1 else np.full(len(distance), np.inf)
        best_cost = float(weights @ nearest)
        swap = None
        for position in range(k):
            without = np.where(owner == position, second, nearest)
            costs = weights @ np.minimum(without[:, None], distance)
            costs[medoids] = np.inf
            candidate = int(np.argmin(costs))
            if costs[candidate] < best_cost - 1e-10:
                best_cost = costs[candidate]
                swap = position, candidate
        if swap is None:
            labels = distance[:, medoids].argmin(axis=1)
            return (labels, medoids) if return_medoids else labels
        medoids[swap[0]] = swap[1]
    raise ValueError("PAM did not converge within 100 swaps")


def _spectral_scores(distance: np.ndarray, maximum: int, neighbours: int):
    # von Luxburg (2007), sections 2.2, 3.2 and 8.3:
    # https://arxiv.org/html/0711.0189v1
    # Binary union k-NN graph, excluding self; L_sym = I - D^-1/2 W D^-1/2.
    # The gap lambda_(K+1) - lambda_K provides evidence for K groups.
    n = len(distance)
    maximum = min(int(maximum), n - 1)
    if maximum < 1:
        return [], ["Spectral: at least two observations are required."]
    neighbours = min(max(1, int(neighbours)), n - 1)
    ranked_distance = distance.copy()
    np.fill_diagonal(ranked_distance, np.inf)
    nearest = np.argsort(ranked_distance, axis=1, kind="stable")[:, :neighbours]
    adjacency = np.zeros((n, n), dtype=float)
    adjacency[np.arange(n)[:, None], nearest] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    components = connected_components(adjacency, directed=False, return_labels=False)
    notes = [
        f"Spectral: binary union graph with {neighbours} nearest neighbours per observation ",
        f"(excluding self), {components} connected components; uses the selected distance metric.",
        "The normalized-Laplacian eigengap is a heuristic, including K=1. ",
        "The neighbour count controls graph connectivity and can change the suggested K.",
    ]
    if components > maximum:
        notes.append(
            f"Spectral: the graph has more components ({components}) than maximum K ({maximum}); "
            "no spectral recommendation. Increase maximum K or the neighbour count."
        )
        return [], notes
    eigenvalues = eigh(laplacian(adjacency, normed=True), eigvals_only=True,
                       subset_by_index=[0, maximum])
    eigenvalues = np.clip(eigenvalues, 0.0, 2.0)
    # Set known zero eigenvalues exactly, avoiding roundoff-driven votes.
    eigenvalues[:components] = 0.0
    gaps = np.maximum(np.diff(eigenvalues), 0.0)
    if not np.any(gaps > 1e-12):
        notes.append("Spectral: no resolvable eigengap within the assessed K range.")
        return [], notes
    return [
        ("Spectral", "Normalized Laplacian eigengap", k, float(gap), "maximize")
        for k, gap in enumerate(gaps, start=1)
    ], notes


def _pair_prediction_strength(labels, predicted, k, weights=None):
    weights = np.ones(len(labels)) if weights is None else weights
    strengths = []
    for group in range(k):
        members = predicted[labels == group]
        if len(members) < 2:
            return 0.0
        w = weights[labels == group]
        mass = np.bincount(members, weights=w, minlength=k)
        squared = np.dot(w, w)
        denominator = w.sum() ** 2 - squared
        if denominator <= 0:
            return 0.0
        strengths.append(float(np.clip((np.dot(mass, mass) - squared) / denominator, 0, 1)))
    return min(strengths)


def _evaluation_kmeans(x, k, seed, weights=None):
    model = KMeans(n_clusters=k, n_init=5, max_iter=200, random_state=seed).fit(x, sample_weight=weights)
    if len(np.unique(model.labels_)) != k:
        raise ValueError(f"K-means could not form {k} distinct clusters")
    if model.n_iter_ >= model.max_iter:
        raise ValueError("K-means reached its iteration limit")
    return model


@lru_cache(maxsize=4)
def _stability_cached(data_bytes, shape, maximum, repeats, weight_bytes=None):
    # Tibshirani & Walther (2005): predict within-test-cluster pairs from
    # training centroids; take the weakest cluster, then average both directions.
    # https://search.r-project.org/CRAN/refmans/fpc/html/prediction.strength.html
    x = np.frombuffer(data_bytes, dtype=np.float64).reshape(shape)
    weights = None if weight_bytes is None else np.frombuffer(weight_bytes, dtype=np.float64)
    rng = np.random.default_rng(2025)
    splits = [rng.permutation(len(x)) for _ in range(repeats)]
    scores, notes = [], []
    for k in range(1, min(maximum, len(x) // 4) + 1):
        values = []
        for repeat, order in enumerate(splits):
            a, b = np.array_split(order, 2)
            if k == 1:
                values.append(1.0)
                continue
            try:
                left = _evaluation_kmeans(x[a], k, 2025 + repeat * 2, None if weights is None else weights[a])
                right = _evaluation_kmeans(x[b], k, 2026 + repeat * 2, None if weights is None else weights[b])
                values.append((
                    _pair_prediction_strength(right.labels_, left.predict(x[b]), k, None if weights is None else weights[b])
                    + _pair_prediction_strength(left.labels_, right.predict(x[a]), k, None if weights is None else weights[a])
                ) / 2)
            except (ValueError, np.linalg.LinAlgError) as error:
                notes.append(f"Stability, K={k}: {error}; this K is not evaluated.")
                break
        if len(values) == repeats:
            scores.append((k, float(np.mean(values)), float(np.std(values, ddof=1))))
    return tuple(scores), tuple(notes)


@lru_cache(maxsize=4)
def _gap_cached(data_bytes, shape, maximum, references, weight_bytes=None):
    # Tibshirani, Walther & Hastie (2001): squared Euclidean within-cluster
    # dispersion; uniform null in the PCA-aligned bounding box.
    # https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/clusGap.html
    x = np.frombuffer(data_bytes, dtype=np.float64).reshape(shape)
    weights = None if weight_bytes is None else np.frombuffer(weight_bytes, dtype=np.float64)
    centred = x - np.average(x, axis=0, weights=weights)
    pca_input = centred if weights is None else centred * np.sqrt(weights[:, None])
    _, _, axes = np.linalg.svd(pca_input, full_matrices=False)
    rotated = centred @ axes.T
    lower, upper = rotated.min(axis=0), rotated.max(axis=0)
    rng = np.random.default_rng(2026)
    nulls = [rng.uniform(lower, upper, size=rotated.shape) for _ in range(references)]
    scores, notes = [], []
    # One additional K is required to apply the rule at the displayed maximum.
    stop = min(maximum + 1, len(x) - 1, len(np.unique(x, axis=0)) - 1)
    for k in range(1, stop + 1):
        try:
            observed = _evaluation_kmeans(rotated, k, 2025, weights).inertia_
            reference = np.array([_evaluation_kmeans(null, k, 3000 + i, weights).inertia_
                                  for i, null in enumerate(nulls)])
            if observed <= 0 or np.any(reference <= 0):
                raise ValueError("zero dispersion makes log dispersion undefined")
            logged = np.log(reference)
            gap = float(logged.mean() - np.log(observed))
            uncertainty = float(logged.std(ddof=0) * np.sqrt(1 + 1 / references))
            if not np.isfinite([gap, uncertainty]).all():
                raise ValueError("non-finite gap statistic")
            scores.append((k, gap, uncertainty))
        except (ValueError, np.linalg.LinAlgError) as error:
            notes.append(f"Gap statistic, K={k}: {error}; the search stops here.")
            break
    return tuple(scores), tuple(notes)


def _resampling_evidence(x, maximum, *, stability=True, gap=True, limit=300,
                         repeats=10, references=20, threshold=0.8, weights=None):
    scores, votes, notes = [], [], []
    if not stability and not gap:
        return scores, votes, notes
    limit = min(max(int(limit), 50), 1000)
    repeats = min(max(int(repeats), 2), 50)
    references = min(max(int(references), 2), 50)
    if len(x) > limit:
        positions = np.sort(np.random.default_rng(2027).choice(len(x), limit, replace=False))
        x = x[positions]
        if weights is not None:
            weights = weights[positions]
    # A common scalar leaves Euclidean K-means and gap comparisons unchanged,
    # while keeping squared distances away from overflow on original-scale data.
    scale = np.max(np.abs(x))
    if not np.isfinite(scale) or scale == 0:
        return [], [], ["Resampling evaluations require finite, varying data."]
    x = np.ascontiguousarray(x / scale, dtype=np.float64)
    maximum = min(maximum, len(x) - 1, len(np.unique(x, axis=0)))
    weight_bytes = None if weights is None else np.ascontiguousarray(_importance_weights(weights)).tobytes()
    key = (x.tobytes(), x.shape, maximum)
    notes.append(f"Resampling evaluations use {len(x)} rows, Euclidean K-means and five starts per fit; "
                 "they evaluate this partition model, not every clustering family. Sampling and any "
                 "standardization are shared across repeats, so these assess the prepared data.")
    if weights is not None:
        notes.append("Weighted resampling uses uniform row sampling/splits and importance-weighted fits; "
                     "pair agreement uses products of weights for distinct observations. Gap references "
                     "retain the same weights and use weighted PCA. These are importance-weight extensions "
                     "of the unweighted evaluations; weights do not increase the independent sample size.")
    if stability:
        if len(x) < 4:
            notes.append("Stability requires at least four observations.")
        else:
            values, messages = _stability_cached(*key, repeats, weight_bytes)
            notes.extend(messages)
            for k, mean, sd in values:
                scores.extend([("Stability", "Prediction strength", k, mean, "threshold"),
                               ("Stability", "Between-split SD", k, sd, "diagnostic")])
            eligible = [k for k, mean, _ in values if mean >= threshold]
            if eligible:
                chosen = max(eligible)
                votes.append(("Stability", "Prediction strength", chosen))
                if chosen == maximum:
                    notes.append("Stability supports the search boundary; larger K values have not been ruled out.")
            notes.append(f"Stability: {repeats} repeated half splits, averaged in both directions; "
                         f"choose the largest K with mean prediction strength ≥ {threshold:.2f}. "
                         "Singleton test clusters score zero. K is capped at floor(sample size / 4); "
                         "K=1 is trivially stable and is the fallback, not evidence of homogeneity.")
    if gap:
        try:
            values, messages = _gap_cached(*key, references, weight_bytes)
            notes.extend(messages)
            for k, value, se in values:
                if k <= maximum:
                    scores.extend([("Gap statistic", "Gap", k, value, "one-SE rule"),
                                   ("Gap statistic", "Reference uncertainty", k, se, "diagnostic")])
            chosen = next((left[0] for left, right in itertools.pairwise(values)
                           if left[0] <= maximum and left[1] >= right[1] - right[2]), None)
            if chosen is not None:
                votes.append(("Gap statistic", "Gap", chosen))
            else:
                notes.append("Gap statistic: no K satisfies the one-SE rule within the available range; "
                             "no vote is cast. Consider increasing maximum K.")
            notes.append(f"Gap statistic: {references} uniform reference samples from the PCA-aligned "
                         "bounding box. Choose the smallest K with Gap(K) ≥ Gap(K+1) − s(K+1). "
                         "An extra K is calculated internally when possible, never displayed beyond the maximum. "
                         "Reference uncertainty is SD(log dispersion) × sqrt(1 + 1/B), not a p-value.")
        except (ValueError, np.linalg.LinAlgError) as error:
            notes.append(f"Gap statistic unavailable: {error}")
    return scores, votes, notes


def _analyse(data: proxy_data, *, maximum=10, limit=500, metric="euclidean",
             method="average", centre="centroids", min_points=10, standardize=True,
             graph_neighbours=10, stability=True, gap=True, evaluation_limit=300,
             stability_repeats=10, gap_references=20, stability_threshold=0.8, use_weights=True):
    notes = []
    predictors = data.role_map.columns_with_role(Role.PREDICTOR)
    weight_columns = data.role_map.columns_with_role(Role.WEIGHTING)
    columns = [c for c in data.columns if c in predictors and c not in weight_columns and not str(c).startswith(Card.SHADOW_PREFIX)
               and pd.api.types.is_numeric_dtype(data.frame[c].dtype)
               and not pd.api.types.is_bool_dtype(data.frame[c].dtype)
               and not pd.api.types.is_complex_dtype(data.frame[c].dtype)]
    raw_weights = None
    if use_weights and weight_columns:
        if len(weight_columns) != 1:
            raise ValueError("Assign exactly one observation-importance column")
        weight_column = next(iter(weight_columns))
        series = data.frame[weight_column]
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_complex_dtype(series.dtype):
            raise ValueError("Observation importance must be numeric")
        raw_weights = series.to_numpy(dtype=float, na_value=np.nan)
        _importance_weights(raw_weights)  # Validate before removing incomplete rows.
    frame = data.frame[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    positive = np.ones(len(frame), dtype=bool) if raw_weights is None else raw_weights > 0
    frame = frame.loc[:, frame.iloc[np.flatnonzero(positive)].nunique(dropna=True) > 1]
    positions = np.flatnonzero(positive & frame.notna().all(axis=1).to_numpy())
    notes.append(f"{len(positions)} of {len(frame)} rows have complete finite numeric predictors and positive importance; {frame.shape[1]} predictors retained.")
    if len(positions) > limit:
        selected = np.sort(np.random.default_rng(2025).choice(len(positions), size=limit, replace=False))
        positions = positions[selected]
    x = frame.iloc[positions].to_numpy(dtype=float)
    weights = None
    if raw_weights is not None and len(positions):
        weights = _importance_weights(raw_weights[positions])
        if np.all(weights == weights[0]):
            weights = None  # Exact compatibility with the original algorithms.
    if x.size:
        scale = np.max(np.abs(x), axis=0)
        safe = x / np.where(scale > 0, scale, 1)
        centre_values = np.average(safe, axis=0, weights=weights)
        spread = np.sqrt(np.average((safe - centre_values) ** 2, axis=0, weights=weights))
        keep = spread > 0
        x = x[:, keep]
        if standardize:
            x = (safe[:, keep] - centre_values[keep]) / spread[keep]
    if raw_weights is not None:
        notes.append(f"Analysis uses {len(x)} rows. Importance column: {weight_column}; zero-weight rows excluded. Weights are normalized to mean one.")
    else:
        notes.append(f"Analysis uses {len(x)} rows. Observation weighting is {'disabled' if not use_weights else 'not assigned'}.")
    notes.append("Only numeric Predictor-role columns enter clustering; weighting and shadow columns are excluded.")
    if weights is not None:
        notes.append("Unequal importance weights: Agglomerative, Divisive, Mixture, Topology and Spectral "
                     "are unavailable and cast no votes. Their current implementations do not support "
                     "observation importance consistently. Turn weighting off to include them.")
        notes.append("Weighted silhouette uses weighted neighbour distances and a weighted observation average. "
                     "CH uses weighted dispersion with actual positive-weight row count for degrees of freedom; "
                     "DB uses weighted centres/scatter and equal contributions from clusters. These extensions "
                     "are invariant to multiplying all importance weights by a constant.")
    scores = []
    votes = []
    def result():
        return {"scores": pd.DataFrame(scores, columns=["Family", "Criterion", "K", "Score", "Direction"]),
                    "votes": pd.DataFrame(votes, columns=["Family", "Criterion", "K"]), "notes": notes}
    if len(x) < 3 or x.shape[1] == 0:
        notes.append("At least three complete rows and one varying numeric predictor are needed for recommendations.")
        return result()
    maximum = min(int(maximum), len(x) - 1, len(np.unique(x, axis=0)))
    if maximum < 2:
        notes.append("The analysis sample has fewer than two distinct observations.")
        return result()
    ks = range(1, maximum + 1)
    distance = squareform(pdist(x, metric="cityblock" if metric == "manhattan" else "euclidean"))
    if not np.isfinite(distance).all():
        notes.append("Distances overflowed; enable standardization or rescale the predictors.")
        return result()
    condensed = squareform(distance)
    if method == "ward" and metric != "euclidean":
        notes.append("Ward linkage requires Euclidean distances; Euclidean is used for the agglomerative hierarchy.")
    tree = linkage(pdist(x) if method == "ward" else condensed, method=method) if weights is None else None
    divisive = _diana(distance, maximum) if weights is None else None
    for family in (("Agglomerative", "Divisive", "Partition") if weights is None else ("Partition",)):
        for k in range(2, maximum + 1):
            try:
                if family == "Agglomerative":
                    labels = cut_tree(tree, n_clusters=[k]).ravel()
                elif family == "Divisive":
                    labels = divisive[k]
                elif centre == "medoids":
                    labels = _pam(distance, k, weights)
                else:
                    labels = KMeans(n_clusters=k, n_init=10, random_state=2025).fit_predict(x, sample_weight=weights)
                if len(np.unique(labels)) != k:
                    continue
                metrics = _weighted_metrics(x, distance, labels, weights) if weights is not None else (("Silhouette", silhouette_score(distance, labels, metric="precomputed"), "maximize"),
                           ("Calinski–Harabasz", calinski_harabasz_score(x, labels), "maximize"),
                           ("Davies–Bouldin", davies_bouldin_score(x, labels), "minimize"))
                for criterion, score, direction in metrics:
                    if np.isfinite(score):
                        scores.append((family, criterion, k, float(score), direction))
            except (ValueError, np.linalg.LinAlgError) as error:
                notes.append(f"{family}, K={k}: {error}")
    # https://sklearn.org/stable/auto_examples/mixture/plot_gmm_selection.html
    # sklearn's BIC is -2 log L + p log n: minimize (mclust reverses the sign).
    # These four covariance structures are a subset of mclust's model family.
    if weights is None:
        for covariance in ("spherical", "diag", "tied", "full"):
            for k in ks:
                try:
                    model = GaussianMixture(n_components=k, covariance_type=covariance,
                                            n_init=2, max_iter=300, random_state=2025).fit(x)
                    if model.converged_:
                        score = model.bic(x)
                        if np.isfinite(score):
                            scores.append(("Mixture", f"BIC ({covariance})", k, float(score), "minimize"))
                    else:
                        notes.append(f"Mixture ({covariance}), K={k}: fit did not converge.")
                except (ValueError, np.linalg.LinAlgError) as error:
                    notes.append(f"Mixture ({covariance}), K={k}: {error}")
        # https://github.com/HuPBA/fast_zero_dimensional_persistence_diagrams
        # H0 Vietoris–Rips finite death times equal single-linkage merge distances.
        # A gap between ascending merges n-k-1 and n-k supports k components.
        deaths = linkage(condensed, method="single")[:, 2]
        for k in range(2, maximum + 1):
            persistence_gap = float(deaths[len(x) - k] - deaths[len(x) - k - 1])
            if persistence_gap > 0:
                scores.append(("Topology", "H0 persistence gap", k, persistence_gap, "maximize"))
        try:
            spectral_scores, spectral_notes = _spectral_scores(distance, maximum, graph_neighbours)
            scores.extend(spectral_scores)
            notes.extend(spectral_notes)
        except (ValueError, np.linalg.LinAlgError) as error:
            notes.append(f"Spectral: eigengap calculation unavailable: {error}")
    for family, criterion in dict.fromkeys((s[0], s[1]) for s in scores):
        candidates = [s for s in scores if s[:2] == (family, criterion)]
        choose = min if candidates[0][4] == "minimize" else max
        winner = choose(candidates, key=lambda s: s[3])
        votes.append((family, criterion, winner[2]))
    extra_scores, extra_votes, extra_notes = _resampling_evidence(
        x, maximum, stability=stability, gap=gap, limit=evaluation_limit,
        repeats=stability_repeats, references=gap_references, threshold=stability_threshold, weights=weights,
    )
    scores.extend(extra_scores)
    votes.extend(extra_votes)
    notes.extend(extra_notes)
    # Explore an adaptive radius range, recording noise separately from K.
    neighbour = min(int(min_points), len(x)) - 1
    radii = np.sort(distance, axis=1)[:, neighbour]
    positive = radii[radii > 0]
    if len(positive):
        epsilons = np.unique(np.linspace(np.quantile(positive, .05), np.quantile(positive, .95), 30))
        for eps in epsilons:
            labels = DBSCAN(eps=float(eps), min_samples=int(min_points), metric="precomputed").fit_predict(distance, sample_weight=weights)
            k = len(set(labels) - {-1})
            noise = float(np.average(labels == -1, weights=weights))
            scores.append(("Density", f"Noise fraction (ε={eps:.5g})", k, noise, "diagnostic"))
            if 1 <= k <= maximum:
                votes.append(("Density", f"Noise fraction (ε={eps:.5g})", k))
    if weights is not None:
        notes.append("DBSCAN uses mean-one importance as neighbourhood mass, so MinPoints refers to "
                     "average-weight observations. Noise fractions are importance-weighted. The radius sweep "
                     "remains a geometric diagnostic based on the sampled observations.")
    density_rows = [row for row in scores if row[0] == "Density"]
    if density_rows:
        excluded = sum(not 1 <= row[2] <= maximum for row in density_rows)
        notes.append(f"Density: {excluded} of {len(density_rows)} radii gave all noise or K outside the assessed range and contribute no support; inspect noise fractions in the table.")
    notes.extend([
        "Silhouette, Calinski–Harabasz and Davies–Bouldin compare K≥2 only. Mixture BIC and the spectral eigengap also evaluate K=1.",
        "K-means and centroid-based scores use Euclidean geometry; the chosen metric applies to silhouette, DIANA, PAM, topology, density and the spectral graph.",
        "The persistence gap and density radius sweep are exploratory heuristics. Density K=0 means all noise, not one group.",
        "Each family contributes at most one total unit of support. Support is agreement among methods, not a probability or proof of inherent groups.",
    ])
    return result()


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = "Number of clusters"
    this.description = "Assess cluster counts using seven clustering families and two resampling evaluations and record a number-of-clusters decision for downstream cluster analysis."

    def front():
        return ui.TagList(
            # ui.input_select("Family", label="Evidence", choices=["Aggregate", *FAMILIES], selected="Aggregate"),
            ui.span("K-th Support barchart", class_="text-primary text-center d-block"),
            shinywidgets.output_widget("Chart", fill=True),
        )
    this.front = front

    def back():
        return ui.div(
            ui.span("Support evidence table", class_="text-primary text-center d-block"),
            ui.output_ui("Table"),
            ui.output_ui("Notes"),
            class_="card-scroll-content",
        )
    this.back = back

    def footer():
        return ui.TagList(
            ui.output_ui("Busy"),
            ui.output_ui("KControl"),
            ui.output_text("Summary"),
        )
    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_checkbox("UseWeights", label="Use assigned observation weighting", value=True,
                guide=this, position="left",
                text="Use one numeric Weighting-role column as relative observation importance. Values must be "
                     "finite and nonnegative, with at least one positive value; zero-weight rows are excluded. "
                     "Weights are normalized to mean one. Unequal weights affect Partition, Density, Stability "
                     "and Gap; the other families are unavailable. Equal weights, or no assigned column, retain "
                     "all families. Disable for equal importance; only predictors still enter clustering."),
            ui.input_slider(
                id="Maximum", label="Largest K to assess", min=2, max=25, value=10,
                guide=this, position="left",
                text="Assess and offer decisions from K=1 through this maximum. Data size and distinct rows may "
                     "reduce the counts that can be evaluated; stability is also capped at a quarter of its "
                     "sample size. Gap may fit K+1 internally to check the boundary. Lowering this maximum "
                     "also caps the recorded decision. More K values increase calculation time."
            ),
            ui.input_checkbox(
                id="Standardize", label="Standardize numeric predictors", value=True,
                guide=this, position="left",
                text="Center each numeric predictor and scale it to unit standard deviation on the analysis "
                     "sample, using importance-weighted means and spreads when enabled. This prevents large "
                     "measurement scales dominating distances. Applies only to this assessment; incoming "
                     "and outgoing values are unchanged. Disable when original relative scales are meaningful."
            ),
            ui.input_select(
                id="Metric", label="Distance metric", choices=["euclidean", "manhattan"],
                guide=this, position="left",
                text="Euclidean is straight-line distance; Manhattan sums absolute coordinate differences. "
                     "Used by silhouette, DIANA, PAM, topology, DBSCAN and the spectral graph, and by "
                     "non-Ward agglomerative linkage. Ward, K-means, centroid-based scores and both "
                     "resampling evaluations retain Euclidean geometry; mixture BIC is unaffected."
            ),
            ui.input_select(
                id="Method", label="Agglomerative linkage", choices=["average", "single", "complete", "ward"],
                guide=this, position="left",
                text="Choose how Agglomerative merges clusters: average uses mean pair distances, single the "
                     "nearest pair, complete the farthest pair, and Ward the increase in within-cluster "
                     "squared dispersion. Single can form chains; complete favors compact groups. Ward "
                     "always uses Euclidean distances. This family is unavailable with unequal importance weights."
            ),
            ui.input_select(
                id="Centre", label="Partition method", choices={"centroids": "K-means", "medoids": "PAM (medoids)"},
                guide=this, position="left",
                text="Choose the Partition family's model. K-means fits Euclidean centers that need not be "
                     "observed rows. PAM chooses actual observations as medoids using the selected distance "
                     "metric and can be slower. Both support observation importance. Stability and Gap "
                     "always evaluate K-means, even when PAM is selected here."
            ),
            ui.input_slider(
                id="GraphNeighbours", label="Spectral graph nearest neighbours", min=1, max=50, value=10, step=1, 
                guide=this, title="Spectral graph connectivity", position="left",
                text="Connect each observation to this many nearest neighbours, excluding itself, "
                     "and make connections undirected. Larger values connect more of the graph "
                     "and may merge local clusters. Capped at the number of sampled rows minus one. "
                     "The Spectral family is unavailable with unequal importance weights.",
            ),
            ui.input_slider(
                id="MinPoints", label="DBSCAN minimum points (including self)", min=2, max=100, value=10,
                guide=this, position="left",
                text="Minimum neighborhood size for a DBSCAN core observation, including itself. With unequal "
                     "importance weights this is neighborhood mass after mean-one normalization. Larger "
                     "values require denser regions and may increase noise. The card sweeps an adaptive "
                     "range of radii; this setting also helps determine that range. It does not fix K."
            ),
            ui.input_checkbox("Stability", label="Evaluate resampling stability (K-means)", value=True,
                guide=this, position="left",
                text="Repeatedly fit K-means on two halves of the evaluation sample and measure pair "
                     "agreement under the other half's model. Uses the weakest cluster in each direction "
                     "and averages across directions and splits. This evaluates K-means only; disable to "
                     "avoid these extra fits. Sampling and standardization are shared across repeats."),
            ui.input_checkbox("Gap", label="Evaluate gap statistic (K-means)", value=True,
                guide=this, position="left",
                text="Compare K-means dispersion with uniform reference samples in a principal-component-"
                     "aligned bounding box. Favor the smallest K satisfying the one-standard-error rule; "
                     "K=1 is eligible. No qualifying K means no vote. Uses K-means only, with importance "
                     "weights retained in reference fits when enabled. Disable to avoid reference fits."),
            ui.input_slider("EvaluationLimit", label="Maximum rows for resampling evaluations",
                            min=50, max=1000, value=300, step=50,
                guide=this, position="left",
                text="Cap the prepared-data sample used by Stability and Gap, independently of the main "
                     "analysis limit. Subsampling is reproducible and uniform without replacement; importance "
                     "is applied in fitting and scoring. Larger samples cost more but may reveal small groups. "
                     "This cap does not limit the other families' work."),
            ui.input_slider("StabilityRepeats", label="Stability split repeats",
                            min=2, max=50, value=10, step=1,
                guide=this, position="left",
                text="Number of reproducible random half splits. Each split fits both halves for every "
                     "eligible K, with five K-means starts per fit. More repeats reduce split-to-split "
                     "uncertainty at approximately proportional cost. The table reports mean prediction "
                     "strength and between-split standard deviation. Used only when Stability is enabled."),
            ui.input_slider("StabilityThreshold", label="Minimum prediction strength",
                            min=0.5, max=0.95, value=0.8, step=0.05,
                guide=this, position="left",
                text="Recommend the largest K whose mean prediction strength meets this threshold. Higher "
                     "values demand stronger agreement. Singleton test clusters score zero; K=1 is trivially "
                     "stable and is a fallback, not proof of one natural group. Changing only this threshold "
                     "reuses cached split scores when available. Used only when Stability is enabled."),
            ui.input_slider("GapReferences", label="Gap reference samples",
                            min=2, max=50, value=20, step=1,
                guide=this, position="left",
                text="Number of simulated null-reference datasets. Each requires K-means fits across K, "
                     "with five starts per fit. More references improve Monte Carlo precision at approximately "
                     "proportional cost; a small count is exploratory. Reference uncertainty feeds the "
                     "one-standard-error rule and is not a p-value. Used only when Gap is enabled."),
            ui.input_slider(
                id="MaxObs", label="Maximum observations to analyze", min=3, max=7, value=4, ticks=True, pre="10^",
                guide=this, position="left",
                text="Analyze at most 10 raised to this value rows (4 means 10,000), using reproducible uniform "
                     "sampling without replacement above the limit. A larger sample may retain small groups "
                     "but increases time and memory sharply: pairwise distance storage grows with the square "
                     "of row count. Resampling evaluations have a separate, lower row cap. Sampling does not "
                     "change outgoing data."
            ),
        )
    this.settings = settings

    def server(input, output, session):
        busy = this.busy()
        previous_incoming = None

        @this.reactable(calc=True)
        def incomingproxy_data():
            try:
                value = this.input_data()
            except SilentOperationInProgressException:
                # This card does not own the upstream task's progress lifecycle.
                # Clear it normally while waiting, avoiding Shiny's persistent
                # output state when hidden/unhidden or refreshed during that task.
                req(False)
            req(value is not None)
            return value


        @output
        @render.ui
        def KControl():
            nonlocal previous_incoming
            maximum = int(input.Maximum())
            incoming = incomingproxy_data().cluster_count or 1
            selected = incoming
            if previous_incoming == incoming:
                with reactive.isolate():
                    try:
                        selected = int(input.K() or incoming)
                    except SilentException:
                        pass
            previous_incoming = incoming
            selected = min(max(1, selected), maximum)
            return ui.input_radio_buttons(
                "K", label=None, choices={str(k): f"K={k}" for k in range(1, maximum + 1)}, 
                selected=str(selected), inline=True, width="100%", 
                guide=this, title="Cluster-count decision", position="top",
                text="Choose K to pass it downstream. K=1 means no cluster segmentation.",
            )

        @this.reactable(calc=True)
        def ChosenCount():
            incoming = incomingproxy_data().cluster_count or 1
            try:
                count = int(input.K() or incoming)
            except SilentException:
                count = incoming
            # Enforce the limit before the browser receives the updated radio group.
            return min(max(1, count), int(input.Maximum()))

        @this.reactable(calc=True)
        @this.settle(seconds=2)
        def Options():
            return {"maximum": int(input.Maximum()), "limit": 10**int(input.MaxObs()),
                        "metric": input.Metric(), "method": input.Method(), "centre": input.Centre(),
                        "use_weights": bool(input.UseWeights()),
                        "min_points": int(input.MinPoints()), "standardize": bool(input.Standardize()),
                        "graph_neighbours": int(input.GraphNeighbours()),
                        "stability": bool(input.Stability()), "gap": bool(input.Gap()),
                        "evaluation_limit": int(input.EvaluationLimit()),
                        "stability_repeats": int(input.StabilityRepeats()),
                        "gap_references": int(input.GapReferences()),
                        "stability_threshold": float(input.StabilityThreshold())}

        @busy.track("Assessing the number of clusters…")
        @this.extended_task
        async def Calculate(data, options):
            return await asyncio.to_thread(_analyse, data, **options)

        @this.reactable()
        def StartAnalysis():
            Calculate.cancel()
            Calculate.invoke(incomingproxy_data().clone(), Options())

        @this.reactable(calc=True)
        def Analysis():
            return Calculate.result()

        @this.reactable(calc=True)
        @this.record_code
        def SelectedData():
            source = incomingproxy_data()
            count = ChosenCount()
            req(count <= len(source))
            return source if count == source.cluster_count else source.with_cluster_count(count)

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render.text
        def Summary():
            return " ".join(Analysis()["notes"][:2])

        @output
        @render.ui
        def Notes():
            return ui.TagList(*(ui.p(note) for note in Analysis()["notes"][2:]))

        @output
        @render_widget
        def Chart():
            maximum = int(input.Maximum())
            categories = [str(k) for k in range(1, maximum + 1)]
            votes = Analysis()["votes"]
            votes = votes[votes.K.between(1, maximum)]
            figure = go.Figure()
            for family in FAMILIES:
                # if input.Family() not in ("Aggregate", family):
                #     continue
                selected = votes[votes.Family == family]
                if selected.empty:
                    continue
                counts = selected.K.value_counts().reindex(range(1, maximum + 1), fill_value=0)
                figure.add_bar(x=categories, y=counts.values / len(selected), name=family, width=0.8)
            if not figure.data:
                figure.add_annotation(text="No usable recommendations for this evidence", showarrow=False)
            figure.update_layout(
                barmode="stack", xaxis_title="Number of clusters (K)",
                yaxis_title="Support (normalized within family)" if bool(this.isFullScreen()) else "Support", 
                template="plotly_white",
                modebar={"orientation": "v"},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#bbd6f8",
                margin={"l": 20, "r": 30, "t": 0, "b": 0},
                xaxis={
                    "type": "category", "categoryorder": "array", "categoryarray": categories,
                    "range": [-0.5, maximum - 0.5], "autorange": False,
                    "tickmode": "array", "tickvals": categories,
                    "fixedrange": True,
                    },
                legend={
                    "orientation": "h",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": 1.1,
                    "yanchor": "top",
                },
                #showlegend=bool(this.isFullScreen()),
            )
            widget = go.FigureWidget(figure)
            widget._config = getattr(widget, "_config", {}) | {"displayModeBar": bool(this.isFullScreen()), "displaylogo": False}
            return widget

        @output
        @render.ui
        def Table():
            return ui.output_data_frame("EvidenceTable")

        @output
        @render.data_frame
        def EvidenceTable():
            analysis = Analysis()
            table = analysis["scores"].copy()
            recommended = set(map(tuple, analysis["votes"].to_numpy()))
            table["Recommended"] = [tuple(row) in recommended for row in table[["Family", "Criterion", "K"]].to_numpy()]
            return render.DataTable(table, width="100%", height="auto")

        session.on_ended(Calculate.cancel)

        return SelectedData

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    this._imports.set(proxy_data(_df=pd.read_csv(Card.ROOT / "data" / "Ass2.csv"), _name="Ass2"))
    this.run()
