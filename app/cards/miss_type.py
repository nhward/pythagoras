from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    # Ensure local modules and packages are resolved from the app directory.
    os.chdir(ROOT)
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

import asyncio
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from cyclic_pandas import is_cyclic
from joblib import Parallel, delayed
from list_pandas import is_list
from module import Module
from plotly.colors import sample_colorscale
from proxy_data import proxy_data
from roles import Role
from shiny import reactive, render, req, ui
from shinywidgets import render_widget
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

#TODO: Restrict columns to predictors only? (maybe drop object, cyclic and list types)

OBS_COUNT = "Obs-count"
LOGGER = logging.getLogger(__name__)
MISSINGNESS_ROW_CLASSES = {
    "Random": "miss-type-random-row",
    "Uncertain": "miss-type-uncertain-row",
    "Patterned": "miss-type-patterned-row",
    "Insufficient data": "miss-type-insufficient-row",
}

@dataclass
class TreeAnalysis:
    """The fitted tree and the quantities used to assess it."""
    model: DecisionTreeClassifier | DecisionTreeRegressor | None
    feature_names: list[str]
    target: str
    task: str
    truth: np.ndarray
    prediction: np.ndarray
    branches: int


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Adjust finite p-values using the Benjamini-Hochberg FDR procedure."""
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    finite = values[valid]
    if finite.size == 0:
        return pd.Series(adjusted, index=p_values.index, dtype=float)
    order = np.argsort(finite)
    ranked = finite[order]
    count = len(ranked)
    corrected = ranked * count / np.arange(1, count + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    restored = np.empty(count, dtype=float)
    restored[order] = np.clip(corrected, 0, 1)
    adjusted[valid] = restored
    return pd.Series(adjusted, index=p_values.index, dtype=float)


def _missing_variables(frame: pd.DataFrame) -> list[str]:
    """Return columns containing at least one missing value."""
    return [column for column in frame.columns if frame[column].isna().any()]


def _is_geometry(series: pd.Series) -> bool:
    return getattr(series.dtype, "name", None) == "geometry"


def _eligible_predictors(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
    excluded: set[str] | None = None,
) -> list[str]:
    """
    Select useful tree predictors: Exclude current target, and shadow variables.
    Also exclude geometry types and objects types that are instances of list, dict,set.
    If a column is categorical, string or object, the cardinality must exceed 50%

    TODO: A big review of this is needed. Possibly just role = predictor if we can be sure the roles have been set sensibly ahead of this card 
    """
    excluded = set() if excluded is None else set(excluded)
    if target is not None:
        excluded.add(target)
    columns = []
    for column in frame.columns:
        if column in excluded or str(column).startswith(Card.SHADOW_PREFIX):
            continue
        series = frame[column]
        if _is_geometry(series):
            continue
        if is_list(series.dtype):
            columns.append(column)
            continue
        if pd.api.types.is_object_dtype(series.dtype):
            try:
                if series.dropna().map(lambda value: isinstance(value, (list, dict, set))).any():
                    continue
            except (TypeError, ValueError):
                continue
        if (
            pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series.dtype)
        ):
            try:
                cardinality = series.nunique(dropna=True)
            except TypeError:
                continue
            if len(frame) and cardinality >= 0.5 * len(frame):
                continue
        columns.append(column)
    return columns


def _mode_or_default(series: pd.Series, default: object) -> object:
    """Return a deterministic non-missing mode, or a supplied default."""
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else default


def _circular_median(series: pd.Series) -> float:
    """Return the observed position minimizing total circular distance."""
    values = series.cyclic.codes().dropna().to_numpy(dtype=float)
    if not values.size:
        return 0.0
    period = float(series.cyclic.period)
    candidates = np.unique(values)
    distances = np.abs(
        np.mod(values[None, :] - candidates[:, None] + period / 2, period)
        - period / 2
    )
    return float(candidates[np.argmin(distances.sum(axis=1))])


def _list_items(donors: list[list[object]]) -> list[object]:
    """Return stable, representation-distinct items found in donor lists."""
    items: dict[tuple[str, str], object] = {}
    for donor in donors:
        for item in donor:
            key = (f"{type(item).__module__}.{type(item).__qualname__}", repr(item))
            items.setdefault(key, item)
    return [items[key] for key in sorted(items)]


def _impute_list_values(
    series: pd.Series,
    donors: list[list[object]],
    *,
    random_state: int,
) -> list[list[object]]:
    """Hot-deck missing lists using reproducibly sampled training donors."""
    rng = np.random.default_rng(random_state)
    result: list[list[object]] = []
    for value in series.array:
        if value is pd.NA or value is None:
            if donors:
                result.append(list(donors[int(rng.integers(len(donors)))]))
            else:
                result.append([])
        else:
            result.append(list(value))
    return result


def _encode_list_values(
    values: list[list[object]],
    items: list[object],
    name: str,
) -> dict[str, pd.Series]:
    """Multi-hot encode imputed lists using training-fold item levels."""
    converted: dict[str, pd.Series] = {}
    for position, item in enumerate(items):
        label = f"{name} contains {item}"
        if label in converted:
            label = f"{label} [{position}]"
        converted[label] = pd.Series(
            [float(item in value) for value in values],
            dtype=float,
        )
    return converted


def _fit_design_matrix(
    frame: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Learn mixed-type encodings and training-fold imputations."""
    converted: dict[str, pd.Series] = {}
    specifications: list[dict[str, object]] = []
    for column in columns:
        series = frame[column]
        name = str(column)
        if is_list(series.dtype):
            donors = [list(value) for value in series.array if value is not pd.NA]
            items = _list_items(donors)
            seed = 2025 + sum(ord(character) for character in name)
            values = _impute_list_values(series, donors, random_state=seed)
            for encoded_name, encoded in _encode_list_values(values, items, name).items():
                converted[encoded_name] = pd.Series(encoded.to_numpy(), index=frame.index)
            specifications.append({
                "column": column,
                "name": name,
                "kind": "list",
                "donors": donors,
                "items": items,
                "seed": seed,
            })
            continue
        if is_cyclic(series.dtype):
            fill = _circular_median(series)
            values = series.cyclic.codes().astype("Float64").astype(float).fillna(fill)
            kind = "cyclic"
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            values = series.astype("int64", copy=False).astype("float64")
            values[series.isna()] = np.nan
            kind = "datetime"
        elif pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
            values = pd.to_numeric(series, errors="coerce").astype("float64")
            kind = "numeric"
        elif pd.api.types.is_bool_dtype(series.dtype):
            fill = bool(_mode_or_default(series, False))
            values = series.fillna(fill).astype("Float64").astype("float64")
            kind = "boolean"
        elif isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered:
            categories = list(series.cat.categories)
            codes = pd.Series(
                pd.Categorical(series, categories=categories, ordered=True).codes,
                index=series.index,
                dtype=float,
            ).replace(-1, np.nan)
            fill = float(_mode_or_default(codes, 0.0))
            values = codes.fillna(fill)
            kind = "ordered"
        else:
            values = series.astype("string")
            fill = str(_mode_or_default(values, ""))
            values = values.fillna(fill)
            categories = sorted(pd.unique(values).tolist())
            specifications.append({
                "column": column,
                "name": name,
                "kind": "categorical",
                "categories": categories,
                "fill": fill,
            })
            for category in categories:
                converted[f"{name} = {category}"] = values.eq(category).astype(float)
            continue
        values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if kind in {"numeric", "datetime"}:
            finite = values.dropna()
            fill = float(finite.median()) if not finite.empty else 0.0
        converted[name] = values.fillna(fill).astype(float)
        specification = {
            "column": column,
            "name": name,
            "kind": kind,
            "fill": fill,
        }
        if kind == "ordered":
            specification["categories"] = categories
        specifications.append(specification)
    matrix = pd.DataFrame(converted, index=frame.index, dtype=float)
    return matrix, specifications


def _transform_design_matrix(
    frame: pd.DataFrame,
    specifications: list[dict[str, object]],
) -> pd.DataFrame:
    """Apply training-fold encodings and imputations to held-out observations."""
    converted: dict[str, pd.Series] = {}
    for specification in specifications:
        column = specification["column"]
        name = str(specification["name"])
        kind = specification["kind"]
        series = frame[column]
        if kind == "list":
            values = _impute_list_values(
                series,
                specification["donors"],
                random_state=int(specification["seed"]),
            )
            encoded_values = _encode_list_values(
                values, specification["items"], name,
            )
            for encoded_name, encoded in encoded_values.items():
                converted[encoded_name] = pd.Series(
                    encoded.to_numpy(), index=frame.index,
                )
            continue
        if kind == "categorical":
            values = series.astype("string").fillna(str(specification["fill"]))
            for category in specification["categories"]:
                converted[f"{name} = {category}"] = values.eq(category).astype(float)
            continue
        if kind == "cyclic":
            values = series.cyclic.codes().astype("Float64").astype(float)
        elif kind == "datetime":
            values = series.astype("int64", copy=False).astype("float64")
            values[series.isna()] = np.nan
        elif kind == "ordered":
            values = pd.Series(
                pd.Categorical(
                    series,
                    categories=specification["categories"],
                    ordered=True,
                ).codes,
                index=series.index,
                dtype=float,
            ).replace(-1, np.nan)
        else:
            values = pd.to_numeric(series, errors="coerce").astype("Float64").astype(float)
        values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        converted[name] = values.fillna(float(specification["fill"])).astype(float)
    return pd.DataFrame(converted, index=frame.index, dtype=float)


def _design_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert mixed Pandas columns using encodings learned from the frame."""
    matrix, _ = _fit_design_matrix(frame, columns)
    return matrix


def _prepare_tree_data(
    frame: pd.DataFrame,
    *,
    target: str,
    add_sequence: bool,
    sample_weight: pd.Series | np.ndarray | None,
    excluded: set[str] | None,
) -> tuple[pd.DataFrame, np.ndarray, str, np.ndarray | None, pd.DataFrame, list[str]]:
    """Construct the common feature, target and weight arrays used by trees."""
    if target != OBS_COUNT and target not in frame.columns:
        raise KeyError(f"Unknown missingness target: {target!r}")
    missing_columns = _missing_variables(frame)
    if target == OBS_COUNT:
        truth = frame.loc[:, missing_columns].isna().sum(axis=1).to_numpy(dtype=float)
        task = "regression"
        target_column = None
    else:
        truth = frame[target].isna().to_numpy(dtype=bool)
        task = "classification"
        target_column = target
    predictors = _eligible_predictors(
        frame,
        target=target_column,
        excluded=excluded,
    )
    working = frame.copy()
    if add_sequence:
        sequence_name = "rownum"
        while sequence_name in working.columns:
            sequence_name = f"_{sequence_name}"
        working[sequence_name] = np.arange(1, len(working) + 1, dtype=float)
        predictors.append(sequence_name)
    matrix = _design_matrix(working, predictors)
    weights = None
    if sample_weight is not None:
        weights = pd.to_numeric(pd.Series(sample_weight, index=frame.index), errors="coerce")
        weights = weights.replace([np.inf, -np.inf], np.nan)
        fill = float(weights.dropna().median()) if weights.notna().any() else 1.0
        weights = weights.fillna(fill).clip(lower=0).to_numpy(dtype=float)
        if not weights.any():
            weights = None
    return matrix, truth, task, weights, working, predictors


def _fit_missingness_tree(
    frame: pd.DataFrame,
    *,
    target: str = OBS_COUNT,
    max_depth: int = 3,
    min_samples_leaf: float = 0.02,
    add_sequence: bool = True,
    sample_weight: pd.Series | np.ndarray | None = None,
    excluded: set[str] | None = None,
    random_state: int = 2025,
) -> TreeAnalysis:
    """Fit the full-data tree used for the descriptive tree chart."""
    matrix, truth, task, weights, _, _ = _prepare_tree_data(
        frame,
        target=target,
        add_sequence=add_sequence,
        sample_weight=sample_weight,
        excluded=excluded,
    )
    # LOGGER.info("Training missing model for %s", target)
    if len(frame) == 0 or matrix.shape[1] == 0:
        prediction = (
            np.full(len(frame), truth.mean() if len(truth) else 0.0)
            if task == "regression"
            else np.full(len(frame), bool(truth.mean() >= 0.5) if len(truth) else False)
        )
        return TreeAnalysis(None, [], target, task, truth, prediction, 0)
    if task == "regression":
        model_class = DecisionTreeRegressor  
    else: 
        model_class = DecisionTreeClassifier
    model = model_class(max_depth = max_depth, min_samples_leaf = float(min_samples_leaf), random_state=random_state)
    model.fit(matrix, truth, sample_weight=weights)
    prediction = model.predict(matrix)
    branches = int(np.count_nonzero(model.tree_.children_left != model.tree_.children_right))
    return TreeAnalysis(
        model=model,
        feature_names=[str(column) for column in matrix.columns],
        target=target,
        task=task,
        truth=truth,
        prediction=np.asarray(prediction),
        branches=branches,
    )


def _fold_balanced_accuracies(
    working: pd.DataFrame,
    predictors: list[str],
    truth: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_depth: int,
    min_samples_leaf: float,
    weights: np.ndarray | None,
    random_state: int,
    include_null: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit and score fresh models in each held-out fold."""
    tree_scores = []
    null_scores = []
    for fold, (train, test) in enumerate(splits):
        train_matrix, specifications = _fit_design_matrix(
            working.iloc[train], predictors,
        )
        test_matrix = _transform_design_matrix(
            working.iloc[test], specifications,
        )
        train_weight = weights[train] if weights is not None else None
        test_weight = weights[test] if weights is not None else None
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + fold,
        )
        model.fit(train_matrix, truth[train], sample_weight=train_weight)
        prediction = model.predict(test_matrix)
        tree_scores.append(balanced_accuracy_score(
            truth[test], prediction, sample_weight=test_weight,
        ))
        if include_null:
            null = DummyClassifier(strategy="prior")
            null.fit(train_matrix, truth[train], sample_weight=train_weight)
            null_prediction = null.predict(test_matrix)
            null_scores.append(balanced_accuracy_score(
                truth[test], null_prediction, sample_weight=test_weight,
            ))
    return (
        np.asarray(tree_scores, dtype=float),
        np.asarray(null_scores, dtype=float) if include_null else None,
    )


def _classification_diagnostics(
    frame: pd.DataFrame,
    *,
    target: str,
    max_depth: int = 4,
    min_samples_leaf: float = 0.02,
    add_sequence: bool = True,
    sample_weight: pd.Series | np.ndarray | None = None,
    excluded: set[str] | None = None,
    cv_folds: int = 5,
    permutations: int = 199,
    random_state: int = 2025,
) -> dict[str, object]:
    """Evaluate a missingness classifier out of sample against a null model."""
    matrix, truth, task, weights, working, predictors = _prepare_tree_data(
        frame,
        target=target,
        add_sequence=add_sequence,
        sample_weight=sample_weight,
        excluded=excluded,
    )
    if task != "classification":
        raise ValueError("A classification target is required")
    missing_count = int(np.count_nonzero(truth))
    observed_count = int(len(truth) - missing_count)
    minority_count = min(missing_count, observed_count)
    result = {
        "Variable": target,
        "CV Balanced Accuracy": np.nan,
        "Null Balanced Accuracy": np.nan,
        "Improvement": np.nan,
        "Fraction Folds Above Null": np.nan,
        "Permutation p-value": np.nan,
        "Missing Count": missing_count,
        "Observed Count": observed_count,
        "CV Folds": 0,
    }
    if matrix.shape[1] == 0 or minority_count < 2:
        return result
    folds = min(max(2, int(cv_folds)), minority_count)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(splitter.split(matrix, truth))
    tree_scores, null_scores = _fold_balanced_accuracies(
        working,
        predictors,
        truth,
        splits,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        weights=weights,
        random_state=random_state,
        include_null=True,
    )
    cv_score = float(np.mean(tree_scores))
    null_score = float(np.mean(null_scores))
    rng = np.random.default_rng(random_state)
    permutation_scores = np.empty(max(0, int(permutations)), dtype=float)
    for index in range(len(permutation_scores)):
        permuted_truth = rng.permutation(truth)
        permutation_splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        permutation_splits = list(permutation_splitter.split(matrix, permuted_truth))
        scores, _ = _fold_balanced_accuracies(
            working,
            predictors,
            permuted_truth,
            permutation_splits,
            max_depth = max_depth,
            min_samples_leaf = min_samples_leaf,
            weights=weights,
            random_state=random_state,
            include_null=False,
        )
        permutation_scores[index] = float(np.mean(scores))
    p_value = (
        float((np.count_nonzero(permutation_scores >= cv_score) + 1) / (len(permutation_scores) + 1))
        if len(permutation_scores)
        else np.nan
    )
    result.update({
        "CV Balanced Accuracy": cv_score,
        "Null Balanced Accuracy": null_score,
        "Improvement": cv_score - null_score,
        "Fraction Folds Above Null": float(np.mean(tree_scores > null_scores)),
        "Permutation p-value": p_value,
        "CV Folds": folds,
    })
    return result


def _fold_r_squared_scores(
    working: pd.DataFrame,
    predictors: list[str],
    truth: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_depth: int,
    min_samples_leaf: float,
    weights: np.ndarray | None,
    random_state: int,
    include_null: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit regression trees and score held-out folds with R-squared."""
    tree_scores = []
    null_scores = []
    for fold, (train, test) in enumerate(splits):
        train_matrix, specifications = _fit_design_matrix(
            working.iloc[train], predictors,
        )
        test_matrix = _transform_design_matrix(
            working.iloc[test], specifications,
        )
        train_weight = weights[train] if weights is not None else None
        test_weight = weights[test] if weights is not None else None
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + fold,
        )
        model.fit(train_matrix, truth[train], sample_weight=train_weight)
        prediction = model.predict(test_matrix)
        tree_scores.append(r2_score(
            truth[test], prediction, sample_weight=test_weight,
        ))
        if include_null:
            null = DummyRegressor(strategy="mean")
            null.fit(train_matrix, truth[train], sample_weight=train_weight)
            null_prediction = null.predict(test_matrix)
            null_scores.append(r2_score(
                truth[test], null_prediction, sample_weight=test_weight,
            ))
    return (
        np.asarray(tree_scores, dtype=float),
        np.asarray(null_scores, dtype=float) if include_null else None,
    )


def _regression_diagnostics(
    frame: pd.DataFrame,
    *,
    max_depth: int = 3,
    min_samples_leaf: float = 0.02,
    add_sequence: bool = True,
    sample_weight: pd.Series | np.ndarray | None = None,
    excluded: set[str] | None = None,
    cv_folds: int = 5,
    permutations: int = 99,
    random_state: int = 2025,
) -> dict[str, object]:
    """Evaluate the observation missing-count regressor against a mean null."""
    matrix, truth, task, weights, working, predictors = _prepare_tree_data(
        frame,
        target=OBS_COUNT,
        add_sequence=add_sequence,
        sample_weight=sample_weight,
        excluded=excluded,
    )
    if task != "regression":
        raise ValueError("A regression target is required")
    result = {
        "CV R-squared": np.nan,
        "Null R-squared": np.nan,
        "Improvement": np.nan,
        "Fraction Folds Above Null": np.nan,
        "Permutation p-value": np.nan,
        "Observations": len(truth),
        "CV Folds": 0,
    }
    if matrix.shape[1] == 0 or len(truth) < 4:
        return result
    # R-squared requires at least two held-out observations in every fold.
    folds = min(max(2, int(cv_folds)), len(truth) // 2)
    if folds < 2:
        return result
    splitter = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(splitter.split(matrix))
    tree_scores, null_scores = _fold_r_squared_scores(
        working,
        predictors,
        truth,
        splits,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        weights=weights,
        random_state=random_state,
        include_null=True,
    )
    cv_score = float(np.mean(tree_scores))
    null_score = float(np.mean(null_scores))
    rng = np.random.default_rng(random_state)
    permutation_scores = np.empty(max(0, int(permutations)), dtype=float)
    for index in range(len(permutation_scores)):
        permuted_truth = rng.permutation(truth)
        scores, _ = _fold_r_squared_scores(
            working,
            predictors,
            permuted_truth,
            splits,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            weights=weights,
            random_state=random_state,
            include_null=False,
        )
        permutation_scores[index] = float(np.mean(scores))
    p_value = (
        float((np.count_nonzero(permutation_scores >= cv_score) + 1) / (len(permutation_scores) + 1))
        if len(permutation_scores)
        else np.nan
    )
    result.update({
        "CV R-squared": cv_score,
        "Null R-squared": null_score,
        "Improvement": cv_score - null_score,
        "Fraction Folds Above Null": float(np.mean(tree_scores > null_scores)),
        "Permutation p-value": p_value,
        "CV Folds": folds,
    })
    return result


def _interpret_regression_model(
    diagnostics: dict[str, object],
    *,
    alpha: float,
    minimum_improvement: float,
    minimum_r_squared: float,
    minimum_fold_fraction: float,
) -> str:
    """Interpret the predictive evidence for observation missing counts."""
    score = float(diagnostics["CV R-squared"])
    if not np.isfinite(score):
        return "Insufficient data"
    statistically_supported = float(diagnostics["Permutation p-value"]) <= float(alpha)
    practically_useful = (
        float(diagnostics["Improvement"]) >= float(minimum_improvement)
        and score >= float(minimum_r_squared)
    )
    stable = (
        float(diagnostics["Fraction Folds Above Null"])
        >= float(minimum_fold_fraction)
    )
    if statistically_supported and practically_useful and stable:
        return "Patterned"
    if (
        float(diagnostics["Improvement"]) > 0
        and (statistically_supported or practically_useful)
    ):
        return "Uncertain"
    return "Random"


def _interpret_missingness_models(
    table: pd.DataFrame,
    *,
    adjust_fdr: bool,
    alpha: float,
    minimum_improvement: float,
    minimum_balanced_accuracy: float,
    minimum_fold_fraction: float,
    minimum_class_count: int,
) -> pd.DataFrame:
    """Add FDR-aware, practical and stability-based interpretations."""
    result = table.copy()
    raw = result["Permutation p-value"]
    result["Adjusted p-value"] = _benjamini_hochberg(raw) if adjust_fdr else raw
    enough = (
        result["Missing Count"].ge(int(minimum_class_count))
        & result["Observed Count"].ge(int(minimum_class_count))
        & result["CV Balanced Accuracy"].notna()
    )
    statistically_supported = result["Adjusted p-value"].le(float(alpha))
    practically_useful = (
        result["Improvement"].ge(float(minimum_improvement))
        & result["CV Balanced Accuracy"].ge(float(minimum_balanced_accuracy))
    )
    stable = result["Fraction Folds Above Null"].ge(float(minimum_fold_fraction))
    patterned = enough & statistically_supported & practically_useful & stable
    potentially_patterned = (
        enough
        & ~patterned
        & result["Improvement"].gt(0)
        & (statistically_supported | practically_useful)
    )
    result["Missingness Type"] = "Random"
    result.loc[potentially_patterned, "Missingness Type"] = "Uncertain"
    result.loc[patterned, "Missingness Type"] = "Patterned"
    result.loc[~enough, "Missingness Type"] = "Insufficient data"
    return result


def _classification_diagnostics_chunk(
    payload: tuple[
        pd.DataFrame,
        list[str],
        dict[str, object],
    ],
) -> list[dict[str, object]]:
    """Fit and diagnose a chunk of targets inside one worker process."""
    frame, targets, options = payload
    rows: list[dict[str, object]] = []
    for target in targets:
        analysis = _fit_missingness_tree(
            frame,
            target=target,
            max_depth=int(options["max_tree_depth"]),
            min_samples_leaf=float(options["min_leaf_samples"]),
            add_sequence=bool(options["add_sequence"]),
            sample_weight=options["sample_weight"],
            excluded=options["excluded"],
            random_state=int(options["random_state"]),
        )
        row = _classification_diagnostics(
            frame,
            target=target,
            max_depth=int(options["max_tree_depth"]),
            min_samples_leaf=float(options["min_leaf_samples"]),
            add_sequence=bool(options["add_sequence"]),
            sample_weight=options["sample_weight"],
            excluded=options["excluded"],
            cv_folds=int(options["cv_folds"]),
            permutations=int(options["permutations"]),
            random_state=int(options["random_state"]),
        )
        row["Branches"] = analysis.branches
        rows.append(row)
    return rows


def _missingness_table(
    frame: pd.DataFrame,
    *,
    targets: list[str] | None = None,
    max_tree_depth: int = 4,
    min_leaf_samples: float = 0.02,
    add_sequence: bool = True,
    sample_weight: pd.Series | np.ndarray | None = None,
    excluded: set[str] | None = None,
    cv_folds: int = 5,
    permutations: int = 199,
    adjust_fdr: bool = True,
    alpha: float = 0.05,
    minimum_improvement: float = 0.05,
    minimum_balanced_accuracy: float = 0.55,
    minimum_fold_fraction: float = 0.8,
    minimum_class_count: int = 20,
    random_state: int = 2025,
    processes: int | None = None,
) -> pd.DataFrame:
    """Cross-validate incomplete variables in target chunks across processes."""
    targets = list(targets)
    columns = [
        "Variable", "Branches", "CV Balanced Accuracy", "Null Balanced Accuracy",
        "Improvement", "Fraction Folds Above Null", "Permutation p-value",
        "Missing Count", "Observed Count", "CV Folds",
    ]
    if not targets:
        empty = pd.DataFrame(columns=columns)
        return _interpret_missingness_models(
            empty,
            adjust_fdr=adjust_fdr,
            alpha=alpha,
            minimum_improvement=minimum_improvement,
            minimum_balanced_accuracy=minimum_balanced_accuracy,
            minimum_fold_fraction=minimum_fold_fraction,
            minimum_class_count=minimum_class_count,
        )
    # Leave two logical CPUs available for Shiny and the rest of the application.
    available_processes = max(1, (os.cpu_count() or 2) - 2)
    if Module.IS_SHINYLIVE:
        process_count = 1
    else:
        process_count = min(
            len(targets),
            available_processes if processes is None else max(1, int(processes))
        )
    options: dict[str, object] = {
        "max_tree_depth": max_tree_depth,
        "min_leaf_samples": min_leaf_samples,
        "add_sequence": add_sequence,
        "sample_weight": sample_weight,
        "excluded": excluded,
        "cv_folds": cv_folds,
        "permutations": permutations,
        "random_state": random_state,
    }
    target_chunks = [
        chunk.tolist()
        for chunk in np.array_split(np.asarray(targets, dtype=object), process_count)
        if len(chunk)
    ]
    payloads = [(frame, chunk, options) for chunk in target_chunks]

    if process_count == 1:
        chunk_rows = [_classification_diagnostics_chunk(payloads[0])]
    else:
        # Loky's reusable process backend is safe to invoke from Shiny's server
        # thread and reuses workers across subsequent reactive evaluations.
        chunk_rows = Parallel(
            n_jobs=process_count,
            backend="loky",
        )(
            delayed(_classification_diagnostics_chunk)(payload)
            for payload in payloads
        )

    rows = [row for chunk in chunk_rows for row in chunk]
    table = pd.DataFrame(rows, columns=columns)
    table = _interpret_missingness_models(
        table,
        adjust_fdr=adjust_fdr,
        alpha=alpha,
        minimum_improvement=minimum_improvement,
        minimum_balanced_accuracy=minimum_balanced_accuracy,
        minimum_fold_fraction=minimum_fold_fraction,
        minimum_class_count=minimum_class_count,
    )
    return table.sort_values(
        ["Adjusted p-value", "CV Balanced Accuracy", "Improvement"],
        ascending=[True, False, False],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def _missingness_row_styles(table: pd.DataFrame) -> list[dict[str, object]]:
    """Return subtle semantic background colours for missingness table rows."""
    if "Missingness Type" not in table.columns:
        return []
    styles: list[dict[str, object]] = []
    for missingness_type, class_name in MISSINGNESS_ROW_CLASSES.items():
        rows = table.index[table["Missingness Type"].eq(missingness_type)].tolist()
        if rows:
            styles.append({
                "rows": rows,
                "class": class_name,
            })
    return styles

def _tree_figure(analysis: TreeAnalysis) -> go.Figure:
    """Draw a fitted scikit-learn decision tree using Plotly annotations."""
    if analysis.model is None or analysis.branches == 0:
        return Card.empty_figure("The selected target has no decision-tree structure")
    tree = analysis.model.tree_
    positions: dict[int, tuple[float, float]] = {}
    next_leaf = 0

    def visit(node: int, depth: int) -> float:
        nonlocal next_leaf
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])

        if left == right:
            x = float(next_leaf)
            next_leaf += 1
        else:
            left_x = visit(left, depth + 1)
            right_x = visit(right, depth + 1)
            x = (left_x + right_x) / 2

        positions[node] = (x, -float(depth))
        return x

    visit(0, 0)
    figure = go.Figure()
    # Combine all branches into one trace.
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for node in range(tree.node_count):
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == right:
            continue
        x0, y0 = positions[node]
        for child in (left, right):
            x1, y1 = positions[child]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={
                "color": "#6c757d",
                "width": 1.5,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    labels: list[str] = []
    impurities: list[float] = []
    for node in range(tree.node_count):
        feature = int(tree.feature[node])
        samples = int(tree.n_node_samples[node])
        if feature >= 0:
            label = (
                f"{analysis.feature_names[feature]} "
                f"≤ {tree.threshold[node]:.3g}"
            )
        elif analysis.task == "classification":
            values = tree.value[node][0]
            total = float(values.sum())
            proportion = (
                float(values[-1] / total)
                if total
                else 0.0
            )
            label = f"Missing = {proportion:.1%}"
        else:
            mean_missing = float(tree.value[node].ravel()[0])
            label = f"Mean missing = {mean_missing:.2f}"
        labels.append(f"{label}<br>n = {samples}")
        impurities.append(float(tree.impurity[node]))
    # Reproduce colorscale="Blues", reversescale=True.
    impurity_values = np.asarray(impurities, dtype=float)
    finite = np.isfinite(impurity_values)
    if finite.any():
        minimum = float(impurity_values[finite].min())
        maximum = float(impurity_values[finite].max())

        if maximum > minimum:
            scaled = (impurity_values - minimum) / (maximum - minimum)
        else:
            scaled = np.full_like(impurity_values, 0.5)
    else:
        scaled = np.full_like(impurity_values, 0.5)
    # Reverse the scale as in the original marker definition.
    scale_positions = 1 - np.clip(scaled, 0, 1)
    node_colours = sample_colorscale(
        "Blues",
        scale_positions.tolist(),
    )
    for node, label, colour, scale_position in zip(
        range(tree.node_count),
        labels,
        node_colours,
        scale_positions,
    ):
        x, y = positions[node]
        # Dark blue backgrounds need white text.
        font_colour = (
            "#ffffff"
            if scale_position >= 0.55
            else "#212529"
        )
        hover_label = (
            f"{label}"
            f"<br>Impurity = {tree.impurity[node]:.3f}"
            f"<br>Weighted n = {tree.weighted_n_node_samples[node]:.1f}"
        )
        figure.add_annotation(
            x=x,
            y=y,
            text=label,
            showarrow=False,
            align="center",
            bgcolor=colour,
            bordercolor="#2c3e50",
            borderwidth=1,
            borderpad=7,
            font={
                "size": 11,
                "color": font_colour,
            },
            # Annotation hover settings
            hovertext=hover_label,
            captureevents=True,
            hoverlabel={
                "bgcolor": "#ffffff",
                "bordercolor": "#2c3e50",
                "font": {
                    "size": 12,
                    "color": "#212529",
                },
            },
        )
    x_values = [position[0] for position in positions.values()]
    y_values = [position[1] for position in positions.values()]
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#bbd6f8",
        margin={
            "l": 25,
            "r": 25,
            "t": 25,
            "b": 25,
        },
        showlegend=False,
        xaxis={
            "visible": False,
            "range": [
                min(x_values) - 0.75,
                max(x_values) + 0.75,
            ],
        },
        yaxis={
            "visible": False,
            "range": [
                min(y_values) - 0.5,
                max(y_values) + 0.5,
            ],
        },
    )
    return figure


def instance():
    """Create the immutable missingness-type card."""
    this = Card(file=__file__, mutable=False)
    this.long_name = "Missingness Type"
    this.description = "This card uses decision trees to assess whether each variable's missingness is random or not."

    def front():
        return ui.navset_bar(
            ui.nav_panel(
                "Obs-count",
                ui.span("Predicting missing-value counts in observations ", class_="text-info text-center d-block"),
                shinywidgets.output_widget(
                    id="Tree",
                    fill=True,
                    guide=this,
                    title="Decision tree",
                    text="A decision tree predicting the selected variable's missingness, or the number of missing values in each observation.",
                    position="left",
                )
            ),
            title = None,
            id = "Target", 
            padding = 0, 
            fillable = True
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.span("Missingness type table", class_="text-primary text-center d-block"),
            ui.output_ui(id="Busy"),
            ui.output_ui(
                id="Table",
                guide=this,
                title="Missingness type table",
                text="Each incomplete variable is classified by whether its decision tree substantially outperforms the null model.",
                position="left",
            )
        )

    this.back = back

    def footer():
        return ui.TagList(
            ui.output_ui(id="Summary"),
       )

    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_checkbox(
                id="AddSeq", label="Add a row-number predictor", value=True,
                guide=this, position="left", text="""
                    "Allows detection of missingness associated with row order or time order (if sequential).
                    <br>This extra predictor allows the models to drift. 
                    It may be that early data had more missing values that more recent data given data 
                    collection has improved over time (assuming the data is in collection order)""",
                
            ),
            ui.input_slider(
                id="MinMissProp", label="Minimum missing proportion", min=0, max=0.5, value=0.05, step=0.01,
                guide=this, text="For a predictor to be considered to have missing values, its missing proportion must exceed this value.", position="left",
            ),
            # ui.input_checkbox(
            #     id="UseWeights",
            #     label="Use observation weights",
            #     value=True,
            #     guide=this,
            #     text="Use any variable assigned with the weighting role as tree observations weights.",
            #     position="left",
            # ),
            ui.input_slider(
                id="CVFolds", label="Cross-validation folds", min=2, max=10, value=5, step=1,
                guide=this, text="Number of stratified held-out folds. This is reduced automatically when the minority class is small.", position="left",
            ),
            ui.input_select(
                id="Permutations", label="Permutation repetitions", selected="99", choices={"99": "100", "199": "200", "499": "500", "999": "1000"},  # because the logic requires reps+1
                guide=this, text="More permutations give a more precise empirical p-value but take longer.", position="left",
            ),
            ui.input_slider(
                id="Alpha", label="Maximum p-value", min=0.01, max=0.10, value=0.05, step=0.01,
                guide=this, position="left", text="""
                    Maximum raw (or adjusted) permutation p-value for a patterned interpretation. Individual-variable classification p-values may be adjusted; the single aggregate regression p-value is unadjusted.
                    <br>Why use several conditions?<br>
                    <ul><li>The permutation p-value controls evidence against predictor–missingness independence.</li>
                    <li>Improvement prevents tiny but statistically significant effects being called meaningful.</li>
                    <li>Absolute balanced accuracy or R-squared prevents good-looking improvements over an unusually poor null score.</li>
                    <li>Fold consistency guards against a result driven by one split.</li>
                    <li>Minimum class counts prevent unstable classification conclusions about very rare missingness.</li></ul>
                    """,
            ),
            ui.input_checkbox(
                id="AdjustFDR", label="Adjust p-values for multiple variables", value=True, 
                guide=this, position="left", text="""
                    Apply Benjamini-Hochberg false-discovery-rate adjustment across the missingness models.<br>
                    Because the hypothesis is testing multiple variables, raw permutation p-values should be adjusted for multiple testing. 
                    The <a href='https://en.wikipedia.org/wiki/False_discovery_rate'>Benjamini–Hochberg false-discovery-rate correction</a> 
                    is appropriate because this is a screening exercise rather than a single confirmatory hypothesis test."
                    """,
            ),
            ui.input_slider(
                id="MinImprovement", label="Minimum score improvement", min=0, max=0.25, value=0.15, step=0.05,
                guide=this, text="Minimum practical improvement in balanced accuracy or R-squared over the matched null model.", position="left",
            ),
            ui.input_slider(
                id="MinBalancedAccuracy", label="Minimum balanced accuracy", min=0.50, max=0.90, value=0.55, step=0.01,
                guide=this, text="Minimum cross-validated balanced accuracy for a patterned classification.", position="left",
            ),
            ui.input_slider(
                id="MinRSquared", label="Minimum R-squared", min=0, max=0.50, value=0.10, step=0.01,
                guide=this, text="Minimum cross-validated R-squared for a patterned missing-count regression.", position="left",
            ),
            ui.input_slider(
                id="MinFoldFraction", label="Minimum fold consistency", min=0.50, max=1, value=0.80, step=0.05,
                guide=this, text="Minimum fraction of held-out folds in which the tree must beat its null model.", position="left",
            ),
            ui.input_numeric(
                id="MinClassCount", label="Minimum missing and observed cases", value=20, min=2, step=1,
                guide=this, text="Both classes must contain at least this many cases before a reliable interpretation is made.", position="left",
            ),
            ui.input_slider(
                id="MaxTreeDepth", label="Maximum Tree depth parameter", min=1, max=5, value=3, step=1,
                guide=this, text="Pre-pruning (for speed) by limiting the depth of the tree hierarchy.", position="left",
            ),
            ui.input_slider(
                id="MinLeafSamples", label="Minimum leaf samples (as a proportion)", min=0.001, max=0.05, value=0.02, step=0.001,
                guide=this, text="Minimum number of samples to justify a leaf node of the tree hierarchy.", position="left",
            ),
            ui.input_slider(
                id="MaxObs", label="Maximum observations to analyse", min=3, max=7, value=4, ticks=True, pre="10^",
                guide=this, text="Limit to number of observations to analyse to ensure responsiveness (logarithmic scale).", position="left",
            ),
        )

    this.settings = settings

    def server(input, output, session):
        busy = this.busy()
        model_cache: OrderedDict[tuple[object, ...], TreeAnalysis] = OrderedDict()
        regression_cache: OrderedDict[tuple[object, ...], dict[str, object]] = OrderedDict()
        cache_owner: proxy_data | None = None
        cache_limit = 64

        def _activate_cache(proxy: proxy_data) -> None:
            """Discard fitted objects whenever the prepared sample changes."""
            nonlocal cache_owner
            if proxy is cache_owner:
                return
            model_cache.clear()
            regression_cache.clear()
            cache_owner = proxy

        def _cache_value(cache: OrderedDict, key: tuple, factory: Callable):
            """Return an LRU-cached value, computing it on the first request."""
            if key in cache:
                value = cache.pop(key)
                cache[key] = value
                return value
            value = factory()
            cache[key] = value
            while len(cache) > cache_limit:
                cache.popitem(last=False)
            return value

        @this.suspendable(calc=True)
        def incomingproxy_data():
            req(this._imports.is_set())
            return this._imports.get()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinMissProp():
            return input.MinMissProp()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinLeafSamples():
            return input.MinLeafSamples()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MaxTreeDepth():
            return input.MaxTreeDepth()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinFoldFraction():
            return input.MinFoldFraction()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinRSquared():
            return input.MinRSquared()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinBalancedAccuracy():
            return input.MinBalancedAccuracy()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinImprovement():
            return input.MinImprovement()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def Alpha():
            return input.Alpha()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def CVFolds():
            return input.CVFolds()

        @this.settle(seconds=2)
        @this.suspendable(calc = True)
        def MaxObs():
            return 10**input.MaxObs()
            
        @this.suspendable(calc=True)
        @this.record_code
        def PreparedData():
            samp =  incomingproxy_data().sample(n=MaxObs(), mode="random", keep_geometry=True)
            return samp

        @this.suspendable(calc=True)
        def MissingVariables():
            minimum_missing_proportion = float(MinMissProp())
            proxy = PreparedData()
            frame = proxy.to_native()
            predictors = proxy.role_map.columns_with_role(Role.PREDICTOR)
            return [
                column
                for column in frame.columns
                if (
                    column in predictors
                    and frame[column].isna().mean() > minimum_missing_proportion
                )
            ]

        PastTabs = reactive.value([])

        @this.suspendable()
        def UpdateChoices():
            for var in PastTabs():
                ui.remove_nav_panel(id = "Target", target=var)
            for var in MissingVariables():
                name = re.sub(r'[^A-Za-z0-9]+', '_', str(var)).strip('_')
                register_tree(output_id=f"{name}__Tree")
                panel = ui.nav_panel(
                    var,
                    ui.span(f"Predicting missing values in {var}", class_="text-primary text-center d-block"),
                    shinywidgets.output_widget(
                        id=f"{name}__Tree",
                        fill=True,
                        guide=this,
                        title="Decision tree",
                        text="A decision tree predicting the selected variable's missingness, or the number of missing values in each observation.",
                        position="left",
                    )
                )
                ui.insert_nav_panel(id = "Target", nav_panel=panel)
                PastTabs.set(MissingVariables())


        def _weighting_column(proxy: proxy_data) -> str | None:
            w = proxy.role_map.columns_with_role(Role.WEIGHTING)
            if len(w) == 0:
                return None
            return list(w)

        def _model_key(
            target: str,
            weighting: str | None,
            excluded: set[str],
        ) -> tuple[object, ...]:
            return (
                target,
                int(MaxTreeDepth()),
                float(MinLeafSamples()),
                bool(input.AddSeq()),
                weighting,
                tuple(sorted(excluded)),
            )

        def _cached_model(proxy: proxy_data, target: str) -> TreeAnalysis:
            _activate_cache(proxy)
            frame = proxy.to_native()
            weighting = _weighting_column(proxy)
            weights = frame[weighting] if weighting is not None else None
            excluded = set(proxy.role_map.columns_with_role(Role.GEOMETRY))
            if weighting is not None:
                excluded.add(weighting)
            this.log.debug(f"Model of {target} sought in cache")
            key = _model_key(target, weighting, excluded)
            return _cache_value(
                model_cache,
                key,
                lambda: _fit_missingness_tree(
                    frame,
                    target=target,
                    max_depth=int(MaxTreeDepth()),
                    min_samples_leaf=float(MinLeafSamples()),
                    add_sequence=bool(input.AddSeq()),
                    sample_weight=weights,
                    excluded=excluded,
                ),
            )

        @this.suspendable(calc=True)
        @this.record_code
        def Model():
            proxy = PreparedData()
            return _cached_model(proxy, input.Target())

        @this.suspendable(calc=True)
        @this.record_code
        def RegressionDiagnostics():
            proxy = PreparedData()
            _activate_cache(proxy)
            frame = proxy.to_native()
            weighting = _weighting_column(proxy)
            weights = frame[weighting] if weighting is not None else None
            excluded = set(proxy.role_map.columns_with_role(Role.GEOMETRY))
            if weighting is not None:
                excluded.add(weighting)
            key = (
                *_model_key(OBS_COUNT, weighting, excluded),
                int(CVFolds()),
                int(input.Permutations()),
            )
            return _cache_value(
                regression_cache,
                key,
                lambda: _regression_diagnostics(
                    frame,
                    max_depth=int(MaxTreeDepth()),
                    min_samples_leaf=float(MinLeafSamples()),
                    add_sequence=bool(input.AddSeq()),
                    sample_weight=weights,
                    excluded=excluded,
                    cv_folds=int(CVFolds()),
                    permutations=int(input.Permutations()),
                ),
            )

        @busy.track("Classifying missingness types…")
        @reactive.extended_task
        async def CalculateTypeTable(
            frame: pd.DataFrame,
            options: dict[str, object],
        ) -> pd.DataFrame:
            """Run the blocking process coordinator outside the reactive graph."""
            return await asyncio.to_thread(
                _missingness_table,
                frame,
                **options,
            )

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @this.suspendable()
        def StartTypeTable():
            """Snapshot reactive values and start a nonblocking table calculation."""
            proxy = PreparedData()
            frame = proxy.to_native()
            weighting = _weighting_column(proxy)
            weights = frame[weighting] if weighting is not None else None
            excluded = set(proxy.role_map.columns_with_role(Role.GEOMETRY))
            if weighting is not None:
                excluded.add(weighting)
            options: dict[str, object] = {
                "targets": MissingVariables(),
                "max_tree_depth": int(MaxTreeDepth()),
                "min_leaf_samples": float(MinLeafSamples()),
                "add_sequence": bool(input.AddSeq()),
                "sample_weight": weights,
                "excluded": excluded,
                "cv_folds": int(CVFolds()),
                "permutations": int(input.Permutations()),
                "adjust_fdr": bool(input.AdjustFDR()),
                "alpha": float(Alpha()),
                "minimum_improvement": float(MinImprovement()),
                "minimum_balanced_accuracy": float(MinBalancedAccuracy()),
                "minimum_fold_fraction": float(MinFoldFraction()),
                "minimum_class_count": int(input.MinClassCount()),
                "processes": 1
            }
            CalculateTypeTable.invoke(frame, options)

        @this.suspendable(calc=True)
        @this.record_code
        def TypeTable():
            """Return the latest table or signal that calculation is in progress."""
            return CalculateTypeTable.result()


        def register_tree(output_id):
            @output(id=output_id)
            @render_widget
            def _():
                figure = _tree_figure(Model())
                figure.update_layout(
                    modebar={"orientation": "v"},
                    modebar_remove=[
                        "select2d", "lasso2d", "toggleHover", "toggleSpikelines",
                        "hoverClosestCartesian", "hoverCompareCartesian",
                    ],
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
        def Tree():
            if not MissingVariables():
                return Card.empty_figure("There are no significantly missing variables")
            figure = _tree_figure(Model())
            figure.update_layout(
                modebar={"orientation": "v"},
                modebar_remove=[
                    "select2d", "lasso2d", "toggleHover", "toggleSpikelines",
                    "hoverClosestCartesian", "hoverCompareCartesian",
                ],
            )
            widget = go.FigureWidget(figure)
            widget._config = getattr(widget, "_config", {}) | {
                "displayModeBar": bool(this.isFullScreen()),
                "displaylogo": False,
                "responsive": True,
            }
            return widget

        @output
        @render.ui
        def Table():
            return ui.output_data_frame(id="Table2")

        @output
        @render.data_frame
        def Table2():
            req(TypeTable() is not None)
            table = TypeTable().copy()
            numeric = [
                "CV Balanced Accuracy", "Null Balanced Accuracy", "Improvement",
                "Fraction Folds Above Null", "Permutation p-value", "Adjusted p-value",
            ]
            table[numeric] = table[numeric].round(3)
            return render.DataTable(
                table,
                width="100%",
                height="98%",
                styles=_missingness_row_styles(table),
            )

        @output
        @render.ui
        def Summary():
            req(this.isFront())
            if not MissingVariables():
                return ui.span(
                    "The data does not contain significantly missing variables.",
                    class_="text-success text-center d-block",
                )
            analysis = Model()
            if analysis.task == "classification":
                table = TypeTable()
                selected = table.loc[table["Variable"] == input.Target()]
                req(not selected.empty)
                row = selected.iloc[0]
                if pd.isna(row["CV Balanced Accuracy"]):
                    return ui.span(
                        f"The sample contains {int(row['Missing Count'])} missing and "
                        f"{int(row['Observed Count'])} observed cases. There is insufficient "
                        "information for cross-validated interpretation.",
                        class_="text-danger text-center d-block",
                    )
                adjustment = (
                    "adjusted"
                    if input.AdjustFDR()
                    else "unadjusted"
                )
                if row['Missingness Type'] == "Patterned":
                    myclass = "text-danger text-center d-block"
                elif row['Missingness Type'] == "Uncertain":  # noqa: SIM114
                    myclass = "text-warning text-center d-block"
                elif row['Missingness Type'] == "Insufficient data":
                    myclass = "text-warning text-center d-block"
                else:
                    myclass = "text-success text-center d-block"
                return ui.TagList(
                    ui.span(
                        f"CV balanced accuracy is {row['CV Balanced Accuracy']:.1%}; "
                        f"null balanced accuracy is {row['Null Balanced Accuracy']:.1%}; "
                        f"improvement is {row['Improvement']:.1%}. The {adjustment} "
                        f"permutation p-value is {row['Adjusted p-value']:.3f}",
                        class_="text-info text-center",
                    ),
                    ui.tags.br(),
                    ui.span(
                        f"Interpretation: {row['Missingness Type']}.",
                        class_=myclass
                    )
                )
            else:
                row = RegressionDiagnostics()
                if pd.isna(row["CV R-squared"]):
                    return ui.span(
                        f"The sample contains {int(row['Observations'])} observations. "
                        "There is insufficient information for cross-validated regression interpretation.",
                        class_="text-warning text-center d-block",
                    )
                interpretation = _interpret_regression_model(
                    row,
                    alpha=float(Alpha()),
                    minimum_improvement=float(MinImprovement()),
                    minimum_r_squared=float(MinRSquared()),
                    minimum_fold_fraction=float(MinFoldFraction()),
                )
                return ui.TagList(
                    ui.span(
                        f"CV R² is {row['CV R-squared']:.1%}; "
                        f"null R² is {row['Null R-squared']:.1%}; "
                        f"improvement is {row['Improvement'] * 100:.1f} percentage points. "
                        "The unadjusted "
                        f"permutation p-value is {row['Permutation p-value']:.3f}",
                        class_="text-info text-center"
                    ),
                    ui.tags.br(),
                    ui.span(
                        f"Interpretation: {interpretation}.",
                        class_="text-info text-center d-block"
                    )
                )
        def cancel_type_table() -> None:
            CalculateTypeTable.cancel()

        session.on_ended(cancel_type_table)

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
