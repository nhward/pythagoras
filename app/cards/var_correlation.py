from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from module import Module
from proxy_data import proxy_data
from roles import Role
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from scipy.stats import norm, rankdata
from shiny import reactive, render, req, ui
from shinywidgets import render_widget
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

SAMPLE_SEED = 1729
DISTANCE_CORRELATION_LIMIT = 2_000
PPS_LIMIT = 10_000
SECTOR_COLOURS = (
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
)


@dataclass(frozen=True)
class CorrelationMethod:
    key: str
    label: str
    title: str
    signed: bool
    directional: bool
    description: str


@dataclass(frozen=True)
class CorrelationAnalysis:
    method: CorrelationMethod
    matrix: pd.DataFrame
    source_observations: int
    analyzed_observations: int
    sampled: bool
    weighted: bool


METHODS = {
    method.key: method
    for method in (
        CorrelationMethod(
            "pearson",
            "Pearson",
            "Pearson correlation",
            True,
            False,
            "Linear association; sensitive to outliers.",
        ),
        CorrelationMethod(
            "spearman",
            "Spearman",
            "Spearman rank correlation",
            True,
            False,
            "Monotonic rank association; less sensitive to outliers than Pearson correlation.",
        ),
        CorrelationMethod(
            "kendall",
            "Kendall",
            "Kendall rank correlation",
            True,
            False,
            "Rank concordance between pairs of observations.",
        ),
        CorrelationMethod(
            "mutual_information",
            "Mutual info",
            "Normalized mutual information",
            False,
            False,
            "A k-nearest-neighbour estimate of general statistical dependence.",
        ),
        CorrelationMethod(
            "distance_correlation",
            "Distance",
            "Distance correlation",
            False,
            False,
            "A non-linear dependence measure based on pairwise distances.",
        ),
        CorrelationMethod(
            "pps",
            "PPS",
            "Predictive power score",
            False,
            True,
            "A decision-tree estimate of out-of-sample improvement; not symmetrical.",
        ),
        CorrelationMethod(
            "latent",
            "Latent",
            "Latent Gaussian rank correlation",
            True,
            False,
            "Pearson correlation after empirical ranks are mapped to normal scores.",
        ),
        CorrelationMethod(
            "xi",
            "Xi",
            "Chatterjee's Xi correlation",
            False,
            True,
            "Dependence of the destination on the source; not symmetrical.",
        ),
    )
}


def _empty_matrix(columns: list[object]) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=columns, columns=columns, dtype=float)


def _finite_pair(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    y = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    x, y, weights = x[keep], y[keep], weights[keep]
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    total = float(weights.sum())
    if total <= 0:
        return np.nan
    x_centred = x - np.average(x, weights=weights)
    y_centred = y - np.average(y, weights=weights)
    denominator = np.sqrt(
        np.sum(weights * x_centred**2) * np.sum(weights * y_centred**2)
    )
    if denominator == 0:
        return np.nan
    return float(np.clip(np.sum(weights * x_centred * y_centred) / denominator, -1, 1))


def _weighted_matrix(
    frame: pd.DataFrame,
    weights: pd.Series,
    *,
    rank: bool,
) -> pd.DataFrame:
    columns = list(frame.columns)
    matrix = _empty_matrix(columns)
    raw_weights = pd.to_numeric(weights, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    for i, left_name in enumerate(columns):
        matrix.iat[i, i] = 1.0
        for j in range(i + 1, len(columns)):
            right_name = columns[j]
            x = pd.to_numeric(frame[left_name], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            y = pd.to_numeric(frame[right_name], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
            keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(raw_weights) & (raw_weights > 0)
            x, y, pair_weights = x[keep], y[keep], raw_weights[keep]
            if rank:
                x = rankdata(x, method="average")
                y = rankdata(y, method="average")
            value = _weighted_correlation(x, y, pair_weights)
            matrix.iat[i, j] = matrix.iat[j, i] = value
    return matrix


def _ordinary_correlation(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    return frame.corr(method=method, min_periods=2).reindex(
        index=frame.columns, columns=frame.columns
    )


def _mutual_information(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns)
    matrix = _empty_matrix(columns)
    for index in range(len(columns)):
        matrix.iat[index, index] = 1.0
    for i, left_name in enumerate(columns):
        for j in range(i + 1, len(columns)):
            x, y = _finite_pair(frame[left_name], frame[columns[j]])
            if len(x) < 4 or np.all(x == x[0]) or np.all(y == y[0]):
                continue
            neighbours = min(3, len(x) - 1)
            forward = mutual_info_regression(
                x.reshape(-1, 1),
                y,
                discrete_features=False,
                n_neighbors=neighbours,
                random_state=SAMPLE_SEED,
            )[0]
            reverse = mutual_info_regression(
                y.reshape(-1, 1),
                x,
                discrete_features=False,
                n_neighbors=neighbours,
                random_state=SAMPLE_SEED,
            )[0]
            estimate = max(0.0, float(forward + reverse) / 2)
            value = float(np.sqrt(1 - np.exp(-2 * estimate)))
            matrix.iat[i, j] = matrix.iat[j, i] = np.clip(value, 0, 1)
    return matrix


def _distance_correlation_pair(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    x_distances = np.abs(x[:, None] - x[None, :])
    y_distances = np.abs(y[:, None] - y[None, :])
    x_centred = (
        x_distances
        - x_distances.mean(axis=0, keepdims=True)
        - x_distances.mean(axis=1, keepdims=True)
        + x_distances.mean()
    )
    y_centred = (
        y_distances
        - y_distances.mean(axis=0, keepdims=True)
        - y_distances.mean(axis=1, keepdims=True)
        + y_distances.mean()
    )
    covariance_squared = float(np.mean(x_centred * y_centred))
    variance_x_squared = float(np.mean(x_centred**2))
    variance_y_squared = float(np.mean(y_centred**2))
    denominator = np.sqrt(variance_x_squared * variance_y_squared)
    if denominator <= 0:
        return np.nan
    return float(np.clip(np.sqrt(max(0.0, covariance_squared) / denominator), 0, 1))


def _distance_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns)
    matrix = _empty_matrix(columns)
    for index in range(len(columns)):
        matrix.iat[index, index] = 1.0
    for i, left_name in enumerate(columns):
        for j in range(i + 1, len(columns)):
            x, y = _finite_pair(frame[left_name], frame[columns[j]])
            value = _distance_correlation_pair(x, y)
            matrix.iat[i, j] = matrix.iat[j, i] = value
    return matrix


def _predictive_power_pair(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 6 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    folds = min(4, max(2, len(x) // 3))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=SAMPLE_SEED)
    model_errors: list[float] = []
    baseline_errors: list[float] = []
    for train, test in splitter.split(x):
        model = DecisionTreeRegressor(random_state=SAMPLE_SEED)
        model.fit(x[train].reshape(-1, 1), y[train])
        prediction = model.predict(x[test].reshape(-1, 1))
        baseline = np.full(len(test), np.median(y[train]), dtype=float)
        model_errors.append(mean_absolute_error(y[test], prediction))
        baseline_errors.append(mean_absolute_error(y[test], baseline))
    model_error = float(np.mean(model_errors))
    baseline_error = float(np.mean(baseline_errors))
    if baseline_error <= 0:
        return np.nan
    return float(np.clip(1 - model_error / baseline_error, 0, 1))


def _predictive_power(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns)
    matrix = _empty_matrix(columns)
    for index in range(len(columns)):
        matrix.iat[index, index] = 1.0
    for i, source in enumerate(columns):
        for j, destination in enumerate(columns):
            if i == j:
                continue
            x, y = _finite_pair(frame[source], frame[destination])
            matrix.iat[i, j] = _predictive_power_pair(x, y)
    return matrix


def _normal_scores(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    probabilities = (ranks - 0.5) / len(values)
    return norm.ppf(probabilities)


def _latent_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns)
    matrix = _empty_matrix(columns)
    for index in range(len(columns)):
        matrix.iat[index, index] = 1.0
    for i, left_name in enumerate(columns):
        for j in range(i + 1, len(columns)):
            x, y = _finite_pair(frame[left_name], frame[columns[j]])
            if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
                continue
            value = float(np.corrcoef(_normal_scores(x), _normal_scores(y))[0, 1])
            matrix.iat[i, j] = matrix.iat[j, i] = np.clip(value, -1, 1)
    return matrix


def _xi_pair(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate Chatterjee's directional xi: dependence of y on x."""
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    rng = np.random.default_rng(SAMPLE_SEED)
    shuffled = rng.permutation(len(x))
    order = shuffled[np.argsort(x[shuffled], kind="stable")]
    ordered_y = y[order]
    right_ranks = rankdata(ordered_y, method="max")
    left_ranks = rankdata(-ordered_y, method="max")
    denominator = 2 * np.sum(left_ranks * (len(y) - left_ranks))
    if denominator <= 0:
        return np.nan
    estimate = 1 - len(y) * np.sum(np.abs(np.diff(right_ranks))) / denominator
    return float(np.clip(estimate, 0, 1))


def _xi_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns)
    matrix = _empty_matrix(columns)
    for index in range(len(columns)):
        matrix.iat[index, index] = 1.0
    for i, source in enumerate(columns):
        for j, destination in enumerate(columns):
            if i == j:
                continue
            x, y = _finite_pair(frame[source], frame[destination])
            matrix.iat[i, j] = _xi_pair(x, y)
    return matrix


def _eligible_columns(data: proxy_data, include_target: bool) -> list[object]:
    roles = {Role.PREDICTOR, Role.TREATMENT}
    if include_target:
        roles.add(Role.TARGET)
    selected = set().union(*(data.role_map.columns_with_role(role) for role in roles))
    frame = data.frame
    return [
        column
        for column in frame.columns
        if column in selected
        and not str(column).startswith(Card.SHADOW_PREFIX)
        and Role.WEIGHTING not in data.role_map.roles_for(column)
        and pd.api.types.is_numeric_dtype(frame[column].dtype)
        and not pd.api.types.is_bool_dtype(frame[column].dtype)
        and frame[column].nunique(dropna=True) > 1
    ]


def _sample_limit(method: str, requested: int) -> int:
    if method == "distance_correlation":
        return min(requested, DISTANCE_CORRELATION_LIMIT)
    if method == "pps":
        return min(requested, PPS_LIMIT)
    return requested


def _analysis_frame(
    data: proxy_data,
    columns: list[object],
    method: str,
    maximum_observations: int,
) -> tuple[pd.DataFrame, pd.Series | None, bool]:
    weighting_columns = [
        column
        for column in data.frame.columns
        if column in data.role_map.columns_with_role(Role.WEIGHTING)
        and not str(column).startswith(Card.SHADOW_PREFIX)
        and pd.api.types.is_numeric_dtype(data.frame[column].dtype)
        and not pd.api.types.is_bool_dtype(data.frame[column].dtype)
    ]
    weight_name = weighting_columns[0] if len(weighting_columns) == 1 else None
    selected = list(columns)
    if method in {"pearson", "spearman"} and weight_name is not None:
        selected.append(weight_name)
    frame = data.frame.loc[:, selected]
    limit = _sample_limit(method, maximum_observations)
    sampled = len(frame) > limit
    if sampled:
        positions = np.sort(
            np.random.default_rng(SAMPLE_SEED).choice(
                len(frame), size=limit, replace=False
            )
        )
        frame = frame.iloc[positions].copy()
    else:
        frame = frame.copy()
    weights = frame.pop(weight_name) if weight_name in frame.columns else None
    return frame, weights, sampled


def _calculate_matrix(
    frame: pd.DataFrame,
    method: str,
    weights: pd.Series | None,
) -> pd.DataFrame:
    if method == "pearson":
        return (
            _weighted_matrix(frame, weights, rank=False)
            if weights is not None
            else _ordinary_correlation(frame, "pearson")
        )
    if method == "spearman":
        return (
            _weighted_matrix(frame, weights, rank=True)
            if weights is not None
            else _ordinary_correlation(frame, "spearman")
        )
    if method == "kendall":
        return _ordinary_correlation(frame, "kendall")
    if method == "mutual_information":
        return _mutual_information(frame)
    if method == "distance_correlation":
        return _distance_correlation(frame)
    if method == "pps":
        return _predictive_power(frame)
    if method == "latent":
        return _latent_correlation(frame)
    if method == "xi":
        return _xi_correlation(frame)
    raise ValueError(f"Unknown correlation method: {method}")


def _analyse_correlation(
    data: proxy_data,
    *,
    method: str,
    include_target: bool,
    maximum_observations: int,
) -> CorrelationAnalysis:
    definition = METHODS[method]
    columns = _eligible_columns(data, include_target)
    source_observations = len(data.frame)
    frame, weights, sampled = _analysis_frame(
        data, columns, method, maximum_observations
    )
    matrix = _calculate_matrix(frame, method, weights)
    return CorrelationAnalysis(
        method=definition,
        matrix=matrix,
        source_observations=source_observations,
        analyzed_observations=len(frame),
        sampled=sampled,
        weighted=weights is not None,
    )


def _matrix_order(matrix: pd.DataFrame, ordering: str) -> list[object]:
    columns = list(matrix.columns)
    if len(columns) < 3 or ordering == "original":
        return columns
    values = np.abs(matrix.to_numpy(dtype=float))
    combined = np.fmax(values, values.T)
    np.fill_diagonal(combined, 1.0)
    if ordering == "mean":
        means = np.nanmean(np.where(np.eye(len(columns), dtype=bool), np.nan, combined), axis=1)
        return [columns[index] for index in np.argsort(-np.nan_to_num(means, nan=-1), kind="stable")]
    similarity = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=0.0)
    distance = np.clip(1 - similarity, 0, 1)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    if ordering == "optimal":
        tree = optimal_leaf_ordering(tree, condensed)
    return [columns[index] for index in leaves_list(tree)]


def _display_matrix(
    analysis: CorrelationAnalysis,
    *,
    threshold: float,
    absolute: bool,
    ordering: str,
) -> pd.DataFrame:
    order = _matrix_order(analysis.matrix, ordering)
    shown = analysis.matrix.loc[order, order].copy()
    values = shown.to_numpy(dtype=float, copy=True)
    if absolute and analysis.method.signed:
        values = np.abs(values)
    below = np.abs(values) < threshold
    np.fill_diagonal(below, False)
    values[below] = 0.0
    return pd.DataFrame(values, index=shown.index, columns=shown.columns)


def _heatmap_figure(
    analysis: CorrelationAnalysis,
    *,
    threshold: float,
    absolute: bool,
    ordering: str,
    full_screen: bool,
) -> go.Figure:
    if analysis.matrix.shape[0] < 2:
        return Card.empty_figure("At least two eligible numeric variables are required")
    shown = _display_matrix(
        analysis,
        threshold=threshold,
        absolute=absolute,
        ordering=ordering,
    )
    signed = analysis.method.signed and not absolute
    labels = [str(value) for value in shown.columns]
    values = shown.to_numpy(dtype=float)
    source = np.broadcast_to(np.asarray(labels)[:, None], values.shape)
    destination = np.broadcast_to(np.asarray(labels)[None, :], values.shape)
    heatmap = go.Heatmap(
        z=values,
        x=labels,
        y=labels,
        customdata=np.dstack((source, destination)),
        colorscale="RdBu_r" if signed else "Blues",
        zmin=-1 if signed else 0,
        zmax=1,
        showscale=full_screen,
        colorbar={
            "title": (
                "Absolute value"
                if absolute and analysis.method.signed
                else "Value"
            )
        },
        hovertemplate=(
            "Source: %{customdata[0]}<br>Destination: %{customdata[1]}"
            "<br>Value: %{z:.4f}<extra></extra>"
        ),
    )
    figure = go.Figure(heatmap)
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#bbd6f8",
        margin={"l": 70, "r": 20, "t": 10, "b": 70},
        xaxis={"title": "Destination" if analysis.method.directional else None, "tickangle": -35},
        yaxis={"title": "Source" if analysis.method.directional else None, "autorange": "reversed"},
    )
    return figure


def _svg_path(x: np.ndarray, y: np.ndarray) -> str:
    points = " ".join(f"L {xv:.6f},{yv:.6f}" for xv, yv in zip(x[1:], y[1:]))
    return f"M {x[0]:.6f},{y[0]:.6f} {points} Z"


def _bezier(radius: float, start: float, stop: float, points: int = 40) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, points)
    p0 = radius * np.array([np.cos(start), np.sin(start)])
    p2 = radius * np.array([np.cos(stop), np.sin(stop)])
    middle = (start + stop) / 2
    control = -0.35 * radius * np.array([np.cos(middle), np.sin(middle)])
    curve = (
        ((1 - t) ** 2)[:, None] * p0
        + (2 * (1 - t) * t)[:, None] * control
        + (t**2)[:, None] * p2
    )
    return curve[:, 0], curve[:, 1]


def _chord_figure(
    analysis: CorrelationAnalysis,
    *,
    threshold: float,
    absolute: bool,
    ordering: str,
    full_screen: bool,
) -> go.Figure:
    if analysis.matrix.shape[0] < 2:
        return Card.empty_figure("At least two eligible numeric variables are required")
    order = _matrix_order(analysis.matrix, ordering)
    matrix = analysis.matrix.loc[order, order]
    values = np.abs(matrix.to_numpy(dtype=float))
    count = len(order)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    sector_arc = 2 * np.pi / count
    inner_radius = 0.82
    gap = min(0.06, sector_arc * 0.12)
    palette = [SECTOR_COLOURS[i % len(SECTOR_COLOURS)] for i in range(count)]
    shapes: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    for i, angle in enumerate(angles):
        theta = np.linspace(angle - sector_arc / 2 + gap, angle + sector_arc / 2 - gap, 50)
        x = np.concatenate((inner_radius * np.cos(theta), np.cos(theta[::-1])))
        y = np.concatenate((inner_radius * np.sin(theta), np.sin(theta[::-1])))
        shapes.append({
            "type": "path",
            "path": _svg_path(x, y),
            "fillcolor": palette[i],
            "line": {"color": "#777", "width": 1},
            "opacity": 0.9,
        })
        annotations.append({
            "x": 1.13 * np.cos(angle),
            "y": 1.13 * np.sin(angle),
            "text": f"<b>{order[i]}</b>",
            "showarrow": False,
            "font": {"size": 11, "color": "#666"},
        })
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_text: list[str] = []
    usable_arc = max(0.02, sector_arc - 2 * gap)
    for i in range(count):
        for j in range(i + 1, count):
            forward, reverse = values[i, j], values[j, i]
            if not np.isfinite(forward):
                forward = 0.0
            if not np.isfinite(reverse):
                reverse = 0.0
            if max(forward, reverse) < threshold:
                continue
            half_i = min(forward, 1.0) * usable_arc / 2
            half_j = min(reverse, 1.0) * usable_arc / 2
            theta_i = np.linspace(angles[i] - half_i, angles[i] + half_i, 25)
            curve_ij = _bezier(inner_radius, angles[i] + half_i, angles[j] - half_j)
            theta_j = np.linspace(angles[j] - half_j, angles[j] + half_j, 25)
            curve_ji = _bezier(inner_radius, angles[j] + half_j, angles[i] - half_i)
            x = np.concatenate((
                inner_radius * np.cos(theta_i),
                curve_ij[0],
                inner_radius * np.cos(theta_j),
                curve_ji[0],
            ))
            y = np.concatenate((
                inner_radius * np.sin(theta_i),
                curve_ij[1],
                inner_radius * np.sin(theta_j),
                curve_ji[1],
            ))
            raw_forward = matrix.iat[i, j]
            raw_reverse = matrix.iat[j, i]
            negative = not absolute and (
                (np.isfinite(raw_forward) and raw_forward < 0)
                or (np.isfinite(raw_reverse) and raw_reverse < 0)
            )
            shapes.append({
                "type": "path",
                "path": _svg_path(x, y),
                "fillcolor": "rgba(214,39,40,0.38)" if negative else "rgba(31,119,180,0.35)",
                "line": {"color": "rgba(100,100,100,0.45)", "width": 0.6},
            })
            midpoint = (angles[i] + angles[j]) / 2
            hover_x.append(0.28 * np.cos(midpoint))
            hover_y.append(0.28 * np.sin(midpoint))
            shown_forward = abs(raw_forward) if absolute else raw_forward
            shown_reverse = abs(raw_reverse) if absolute else raw_reverse
            hover_text.append(
                f"{order[i]} → {order[j]}: {shown_forward:.4f}<br>"
                f"{order[j]} → {order[i]}: {shown_reverse:.4f}"
            )
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=hover_x,
        y=hover_y,
        mode="markers",
        marker={"size": 12, "color": "rgba(0,0,0,0.01)"},
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    if full_screen and not absolute:
        for name, colour in (
            ("Positive association", "rgba(31,119,180,0.70)"),
            ("Negative association", "rgba(214,39,40,0.70)"),
        ):
            figure.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 12, "color": colour},
                name=name,
                hoverinfo="skip",
                showlegend=True,
            ))
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=shapes,
        annotations=annotations,
        xaxis={"visible": False, "range": [-1.25, 1.25], "fixedrange": True},
        yaxis={"visible": False, "range": [-1.25, 1.25], "scaleanchor": "x", "fixedrange": True},
        margin={"l": 5, "r": 5, "t": 5, "b": 5},
        showlegend=full_screen and not absolute,
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.02,
            "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0.75)",
        },
    )
    return figure


def _values_table(analysis: CorrelationAnalysis, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    matrix = analysis.matrix
    for i, source in enumerate(matrix.index):
        destinations = range(len(matrix.columns)) if analysis.method.directional else range(i + 1, len(matrix.columns))
        for j in destinations:
            if i == j:
                continue
            value = matrix.iat[i, j]
            if not np.isfinite(value) or abs(value) < threshold:
                continue
            rows.append({
                "Source": str(source),
                "Destination": str(matrix.columns[j]),
                "Value": float(value),
            })
    result = pd.DataFrame(rows, columns=["Source", "Destination", "Value"])
    if not result.empty:
        result = result.iloc[np.argsort(-np.abs(result["Value"].to_numpy()), kind="stable")]
        result["Value"] = result["Value"].round(4)
        result.reset_index(drop=True, inplace=True)
    return result


def instance():
    """Create the immutable variable-correlation card."""
    this = Card(file=__file__, mutable=False)
    this.long_name = "Variable correlation"
    this.description = (
        "This card compares numeric variables using linear, rank, non-linear, "
        "predictive, and directional measures of association."
    )

    def _panel(method: CorrelationMethod):
        return ui.nav_panel(
            method.label,
            ui.div(
                ui.span(
                    method.title,
                    class_="text-primary text-center d-block",
                ),
                shinywidgets.output_widget(
                    id=f"Chart_{method.key}",
                    fill=True,
                    guide=this,
                    title=method.title,
                    text=method.description,
                    position="left",
                ),
                class_="html-fill-container html-fill-item",
                style="width:100%; height:100%;",
            ),
            value=method.key,
        )

    def front():
        return ui.navset_bar(
            *(_panel(method) for method in METHODS.values()),
            title=None,
            id="CorrType",
            selected="pearson",
            padding=0,
            fillable=True,
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.output_ui(id="TableTitle"),
            ui.output_ui(
                id="Table",
                guide=this,
                title="Association values",
                text=(
                    "Lists values at or above the selected magnitude threshold. "
                    "Directional methods retain both source-to-destination directions."
                ),
                position="left",
            ),
        )

    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(id="Busy"),
            ui.output_ui(
                id="Check",
                guide=this,
                title="Correlation analysis summary",
                text="Reports the variables, observations, weighting, and directionality used by the selected method.",
                position="top",
            ),
            class_="text-center",
        )

    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_radio_buttons(
                id="Style", label="Chart style", choices={"heatmap": "Heat map", "chord": "Chord"}, selected="chord",
                guide=this, position="left", text="Switches between a heatmap chart and a chord chart.",
            ),
            ui.input_slider(
                id="Threshold", label="Association threshold", min=0, max=1, value=0.25, step=0.01,
                guide=this, position="left",
                text="""
                Values below this threshold (by absolute magnitude) are hidden from the charts and tables.
                This entirely subjective and is intended to hide associations that are indistinguishable from zero.
                This may also be a mechanism to expose the large associations.
                """
            ),
            ui.input_checkbox(
                id="IncludeTarget", label="Include target variables", value=False,
                guide=this, position="left", text="Adds numeric, nonconstant target variables to predictors and treatment variables.",
            ),
            ui.input_checkbox(
                id="Absolute", label="Show absolute values", value=False,
                guide=this, position="left",
                text="Removes the sign from signed measures in both chart styles. Chord widths always use magnitude; without absolute values, ribbon colours distinguish positive and negative associations.",
            ),
            ui.input_select(
                id="Ordering", label="Variable ordering", selected="optimal",
                choices={
                    "optimal": "Optimal leaf ordering",
                    "clustered": "Hierarchical clustering",
                    "mean": "Mean association",
                    "original": "Original column order",
                },
                guide=this, position="left",
                text="Changes presentation only. Assymmetric matrices are combined with their transpose solely to calculate a common row and column order."
            ),
            ui.input_slider(
                id="MaxObs", label="Maximum observations to analyze", min=2, max=5, value=3, ticks=True, pre="10^",
                guide=this, position="left",
                text="Above 10^n rows the chart uses a deterministic random sample. Distance correlation is capped at 2,000 and PPS at 10,000 observations for computational safety."
            ),
        )

    this.settings = settings

    def server(input, output, session):
        busy = this.busy()

        @this.suspendable(calc=True)
        def incomingproxy_data():
            return this.input_data()

        @this.settle(seconds=0.4)
        @this.suspendable(calc=True)
        def AnalysisOptions():
            method = input.CorrType() or "pearson"
            req(method in METHODS)
            return {
                "method": method,
                "include_target": bool(input.IncludeTarget()),
                "maximum_observations": 10 ** int(input.MaxObs()),
            }

        @this.settle(seconds=0.3)
        @this.suspendable(calc=True)
        def DisplayOptions():
            return {
                "style": str(input.Style()),
                "threshold": float(input.Threshold()),
                "absolute": bool(input.Absolute()),
                "ordering": str(input.Ordering()),
            }

        @busy.track("Calculating variable associations…")
        @reactive.extended_task
        async def Calculate(data: proxy_data, options: dict[str, object]):
            return await asyncio.to_thread(_analyse_correlation, data, **options)

        @this.suspendable()
        def StartAnalysis():
            Calculate.invoke(incomingproxy_data(), AnalysisOptions())

        @this.suspendable(calc=True)
        @this.record_code
        def Analysis():
            return Calculate.result()

        @output
        @render.ui
        def Busy():
            return busy.ui()

        def chart(method: str):
            analysis = Analysis()
            req(analysis.method.key == method)
            options = DisplayOptions()
            if options["style"] == "chord":
                figure = _chord_figure(
                    analysis,
                    threshold=options["threshold"],
                    absolute=options["absolute"],
                    ordering=options["ordering"],
                    full_screen=bool(this.isFullScreen()),
                )
            else:
                figure = _heatmap_figure(
                    analysis,
                    threshold=options["threshold"],
                    absolute=options["absolute"],
                    ordering=options["ordering"],
                    full_screen=bool(this.isFullScreen()),
                )
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#bbd6f8",
                modebar={"orientation": "v"},
                margin={"l": 200, "r": 200, "t": 0, "b": 0} if this.isFullScreen() else {"l": 100, "r": 100, "t": 0, "b": 0}
            )
            widget = go.FigureWidget(figure)
            widget._config = getattr(widget, "_config", {}) | {
                "displayModeBar": bool(this.isFullScreen()),
                "displaylogo": False,
                "responsive": True,
            }
            return widget

        @output
        @render_widget
        def Chart_pearson():
            return chart("pearson")

        @output
        @render_widget
        def Chart_spearman():
            return chart("spearman")

        @output
        @render_widget
        def Chart_kendall():
            return chart("kendall")

        @output
        @render_widget
        def Chart_mutual_information():
            return chart("mutual_information")

        @output
        @render_widget
        def Chart_distance_correlation():
            return chart("distance_correlation")

        @output
        @render_widget
        def Chart_pps():
            return chart("pps")

        @output
        @render_widget
        def Chart_latent():
            return chart("latent")

        @output
        @render_widget
        def Chart_xi():
            return chart("xi")

        @output
        @render.ui
        def TableTitle():
            analysis = Analysis()
            qualifier = "directional " if analysis.method.directional else ""
            return ui.span(
                f"{analysis.method.title} — {qualifier}values",
                class_="text-primary text-center d-block",
            )

        @output
        @render.ui
        def Table():
            req(Analysis() is not None)
            return ui.output_data_frame(id="Table2")

        @output
        @render.data_frame
        def Table2():
            analysis = Analysis()
            table = _values_table(analysis, DisplayOptions()["threshold"])
            return render.DataTable(table, width="100%", height="98%")

        @output
        @render.ui
        def Check():
            analysis = Analysis()
            variables = analysis.matrix.shape[0]
            if variables < 2:
                return ui.span(
                    f"Only {variables} eligible numeric variable{' is' if variables == 1 else 's are'} available; at least two are required.",
                    class_="text-warning",
                )
            basis = (
                f"a deterministic sample of {analysis.analyzed_observations:,} from "
                f"{analysis.source_observations:,} observations"
                if analysis.sampled
                else f"all {analysis.source_observations:,} observations"
            )
            details = [
                f"{variables} variables",
                basis,
                "directional matrix" if analysis.method.directional else "symmetric matrix",
            ]
            if analysis.weighted:
                details.append("observation weights applied")
            return ui.span("; ".join(details) + ".", class_="text-info")

        session.on_ended(Calculate.cancel)
        return incomingproxy_data

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
