from __future__ import annotations

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

import asyncio

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from cyclic_pandas import is_cyclic
from geometry_pandas import is_geometry
from joblib import Parallel, delayed
from list_pandas import is_list
from module import Module
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import reactive, render, req, ui
from shinywidgets import render_widget
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

METHOD_LABELS = {
    "simple": "Median or mode",
    "knn": "Nearest neighbours",
    "iterative": "Iterative prediction",
}

IMPUTATION_ROW_CLASSES = {
    "Strong": "miss-impute-strong-row",
    "Weak": "miss-impute-weak-row",
    "Too few observed values": "miss-impute-insufficient-row",
}


def _kind(series: pd.Series) -> str:
    if is_cyclic(series.dtype):
        return "cyclic"
    if is_list(series.dtype):
        return "list"
    if is_geometry(series.dtype):
        return "geometry"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "ordered" if series.cat.ordered else "categorical"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    if pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype):
        return "categorical"
    return "unsupported"


def _eligible_columns(frame: pd.DataFrame, role_map: RoleMap) -> tuple[list[str], dict[str, str]]:
    predictors = role_map.columns_with_role(Role.PREDICTOR)
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    for column in frame.columns:
        if not frame[column].isna().any():
            continue
        if str(column).startswith(Card.SHADOW_PREFIX):
            excluded[column] = "Shadow variable"
            continue
        if column not in predictors:
            excluded[column] = "Not assigned the predictor role"
            continue
        kind = _kind(frame[column])
        if kind in {"cyclic", "list", "geometry", "unsupported"}:
            excluded[column] = f"{kind.title()} predictors are not imputed"
            continue
        if frame[column].notna().sum() == 0:
            excluded[column] = "No observed values are available"
            continue
        eligible.append(column)
    return eligible, excluded


def _predictor_columns(frame: pd.DataFrame, role_map: RoleMap) -> list[str]:
    """Return supported predictor columns that may inform an imputation."""
    predictors = role_map.columns_with_role(Role.PREDICTOR)
    return [
        column for column in frame.columns
        if column in predictors
        and not str(column).startswith(Card.SHADOW_PREFIX)
        and _kind(frame[column]) not in {"cyclic", "list", "geometry", "unsupported"}
        and frame[column].notna().any()
    ]


def _simple_fill(series: pd.Series) -> tuple[pd.Series, str]:
    """Impute one Series using scikit-learn's SimpleImputer."""
    kind = _kind(series)
    missing = series.isna()
    if not missing.any():
        return series.copy(), "Unchanged"
    if kind in {"numeric", "datetime", "ordered"}:
        values, _ = _encode_continuous(series)
        imputed = SimpleImputer(strategy="median").fit_transform(
            values.reshape(-1, 1)
        )[:, 0]
        return _restore_continuous(series, imputed), "Median"
    values = series.astype(object).where(series.notna(), np.nan).to_numpy()
    imputed = SimpleImputer(strategy="most_frequent").fit_transform(
        values.reshape(-1, 1)
    )[:, 0]
    result = series.copy()
    result.loc[missing] = imputed[missing.to_numpy()]
    return result, "Mode"


def _encode_continuous(series: pd.Series) -> tuple[np.ndarray, dict[str, object]]:
    kind = _kind(series)
    if kind == "ordered":
        values = series.cat.codes.to_numpy(dtype=float)
        values[values < 0] = np.nan
        return values, {"kind": kind, "categories": list(series.cat.categories)}
    if kind == "datetime":
        values = series.astype("datetime64[ns]").astype("int64").to_numpy(dtype=float)
        values[series.isna().to_numpy()] = np.nan
        return values, {"kind": kind}
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    values[~np.isfinite(values)] = np.nan
    return values, {"kind": kind}


def _restore_continuous(original: pd.Series, values: np.ndarray) -> pd.Series:
    kind = _kind(original)
    result = original.copy()
    missing = original.isna().to_numpy()
    if kind == "ordered":
        positions = np.rint(values[missing]).astype(int)
        positions = np.clip(positions, 0, len(original.cat.categories) - 1)
        result.loc[original.isna()] = [original.cat.categories[value] for value in positions]
    elif kind == "datetime":
        restored = pd.Series(
            pd.to_datetime(np.rint(values[missing]).astype("int64"), unit="ns")
        ).astype(original.dtype)
        result.loc[original.isna()] = restored.to_numpy()
    else:
        result.loc[original.isna()] = values[missing]
    return result


class ImputationStep(TransformerMixin, BaseEstimator):
    """A DataFrame-preserving sklearn imputation step for Pythagoras."""

    def __init__(
        self,
        eligible: tuple[str, ...],
        predictors: tuple[str, ...],
        method: str,
        neighbours: int = 5,
        iterations: int = 10,
        seed: int = 2025,
    ):
        self.eligible = eligible
        self.predictors = predictors
        self.method = method
        self.neighbours = neighbours
        self.iterations = iterations
        self.seed = seed

    @staticmethod
    def _fit_simple(series: pd.Series) -> SimpleImputer:
        kind = _kind(series)
        if kind in {"numeric", "datetime", "ordered"}:
            values, _ = _encode_continuous(series)
            return SimpleImputer(
                strategy="median", keep_empty_features=True,
            ).fit(values.reshape(-1, 1))
        values = series.astype(object).where(series.notna(), np.nan).to_numpy()
        return SimpleImputer(
            strategy="most_frequent", keep_empty_features=True,
        ).fit(values.reshape(-1, 1))

    @staticmethod
    def _apply_simple(series: pd.Series, imputer: SimpleImputer) -> pd.Series:
        kind = _kind(series)
        if kind in {"numeric", "datetime", "ordered"}:
            values, _ = _encode_continuous(series)
            filled = imputer.transform(values.reshape(-1, 1))[:, 0]
            return _restore_continuous(series, filled)
        values = series.astype(object).where(series.notna(), np.nan).to_numpy()
        filled = imputer.transform(values.reshape(-1, 1))[:, 0]
        result = series.copy()
        missing = series.isna().to_numpy()
        result.loc[series.isna()] = filled[missing]
        return result

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ImputationStep requires a pandas DataFrame")
        missing = [column for column in self.predictors if column not in X.columns]
        if missing:
            raise ValueError(f"Required variables are absent: {missing}")

        self.simple_imputers_ = {}
        self.unlearnable_ = []
        for column in self.eligible:
            if column not in X.columns or not X[column].notna().any():
                self.unlearnable_.append(column)
                continue
            self.simple_imputers_[column] = self._fit_simple(X[column])
        self.continuous_ = [
            column for column in self.predictors
            if _kind(X[column]) in {"numeric", "datetime", "ordered"}
            and X[column].notna().any()
        ]
        self.targets_ = [
            column for column in self.eligible if column in self.continuous_
        ]
        self.methods_ = {
            column: (
                "Median" if _kind(X[column]) in {"numeric", "datetime", "ordered"}
                else "Mode"
            )
            for column in self.eligible
            if column in X.columns and column not in self.unlearnable_
        }

        self.continuous_imputer_ = None
        if self.method in {"knn", "iterative"} and self.targets_ and len(X) >= 2:
            matrix = np.column_stack([
                _encode_continuous(X[column])[0] for column in self.continuous_
            ])
            self.centres_ = np.nanmedian(matrix, axis=0)
            self.centres_[~np.isfinite(self.centres_)] = 0.0
            self.scales_ = np.nanstd(matrix, axis=0)
            self.scales_[~np.isfinite(self.scales_) | (self.scales_ == 0)] = 1.0
            scaled = (matrix - self.centres_) / self.scales_
            if self.method == "knn":
                count = min(max(1, int(self.neighbours)), max(1, len(X) - 1))
                self.continuous_imputer_ = KNNImputer(
                    n_neighbors=count,
                    weights="distance",
                    keep_empty_features=True,
                ).fit(scaled)
                label = f"Nearest neighbours ({count})"
            else:
                count = max(1, int(self.iterations))
                self.continuous_imputer_ = IterativeImputer(
                    max_iter=count,
                    random_state=int(self.seed),
                    initial_strategy="median",
                    skip_complete=False,
                    keep_empty_features=True,
                ).fit(scaled)
                label = f"Iterative prediction ({count} iterations)"
            for column in self.targets_:
                self.methods_[column] = label

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "simple_imputers_"):
            raise RuntimeError("ImputationStep must be fitted before use")
        result = X.copy()
        for column, imputer in self.simple_imputers_.items():
            if column not in result.columns:
                raise ValueError(f"Required variable {column!r} is absent")
            if column not in self.targets_ or self.continuous_imputer_ is None:
                result[column] = self._apply_simple(result[column], imputer)

        if self.continuous_imputer_ is not None:
            matrix = np.column_stack([
                _encode_continuous(X[column])[0] for column in self.continuous_
            ])
            scaled = (matrix - self.centres_) / self.scales_
            restored = (
                self.continuous_imputer_.transform(scaled) * self.scales_
                + self.centres_
            )
            for position, column in enumerate(self.continuous_):
                if column in self.targets_:
                    result[column] = _restore_continuous(
                        X[column], restored[:, position],
                    )
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            self.feature_names_in_ if input_features is None else input_features,
            dtype=object,
        )


@dataclass
class ImputationResult:
    frame: pd.DataFrame
    summary: pd.DataFrame
    evaluation: pd.DataFrame
    transformer: ImputationStep


def _simple_impute(frame: pd.DataFrame, eligible: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    result = frame.copy()
    methods = {}
    for column in eligible:
        result[column], methods[column] = _simple_fill(result[column])
    return result, methods


def _continuous_impute(
    frame: pd.DataFrame,
    eligible: list[str],
    predictors: list[str],
    imputer,
    method_name: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply a numeric sklearn imputer, with SimpleImputer categorical fallback."""
    result, methods = _simple_impute(frame, eligible)
    continuous = [
        column for column in predictors
        if _kind(frame[column]) in {"numeric", "datetime", "ordered"}
        and frame[column].notna().any()
    ]
    targets = [column for column in eligible if column in continuous]
    if not targets or len(frame) < 2:
        return result, methods
    arrays = [_encode_continuous(frame[column])[0] for column in continuous]
    matrix = np.column_stack(arrays)
    centres = np.nanmedian(matrix, axis=0)
    scales = np.nanstd(matrix, axis=0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    scaled = (matrix - centres) / scales
    restored = imputer.fit_transform(scaled) * scales + centres
    for position, column in enumerate(continuous):
        if column in targets:
            result[column] = _restore_continuous(frame[column], restored[:, position])
            methods[column] = method_name
    return result, methods


def _knn_impute(frame: pd.DataFrame, eligible: list[str], predictors: list[str], neighbours: int) -> tuple[pd.DataFrame, dict[str, str]]:
    count = min(max(1, int(neighbours)), max(1, len(frame) - 1))
    return _continuous_impute(
        frame, eligible, predictors,
        KNNImputer(n_neighbors=count, weights="distance"),
        f"Nearest neighbours ({count})",
    )


def _iterative_impute(frame: pd.DataFrame, eligible: list[str], predictors: list[str], iterations: int, seed: int) -> tuple[pd.DataFrame, dict[str, str]]:
    count = max(1, int(iterations))
    return _continuous_impute(
        frame, eligible, predictors,
        IterativeImputer(
            max_iter=count,
            random_state=seed,
            initial_strategy="median",
            skip_complete=False,
        ),
        f"Iterative prediction ({count} iterations)",
    )


def _impute_frame(frame: pd.DataFrame, eligible: list[str], predictors: list[str], method: str, neighbours: int, iterations: int, seed: int) -> tuple[pd.DataFrame, dict[str, str]]:
    if method == "knn":
        return _knn_impute(frame, eligible, predictors, neighbours)
    if method == "iterative":
        return _iterative_impute(frame, eligible, predictors, iterations, seed)
    return _simple_impute(frame, eligible)


def _score(truth: pd.Series, prediction: pd.Series, kind: str) -> tuple[str, float]:
    if kind in {"numeric", "datetime", "ordered"}:
        if kind == "datetime":
            actual = truth.astype("datetime64[ns]").astype("int64").to_numpy(dtype=float) / 86_400_000_000_000
            predicted = prediction.astype("datetime64[ns]").astype("int64").to_numpy(dtype=float) / 86_400_000_000_000
            return "MAE (days)", float(np.mean(np.abs(actual - predicted)))
        if kind == "ordered":
            categories = truth.dtype.categories
            actual = pd.Categorical(truth, categories=categories, ordered=True).codes
            predicted = pd.Categorical(prediction, categories=categories, ordered=True).codes
            return "Ordinal MAE", float(np.mean(np.abs(actual - predicted)))
        actual = pd.to_numeric(truth).to_numpy(dtype=float)
        predicted = pd.to_numeric(prediction).to_numpy(dtype=float)
        return "MAE", float(np.mean(np.abs(actual - predicted)))
    return "Accuracy", float(np.mean(truth.astype("string").to_numpy() == prediction.astype("string").to_numpy()))


def _random_donor_baseline(
    trial: pd.Series,
    held_positions: np.ndarray,
    rng: np.random.Generator,
) -> pd.Series:
    """Fill held positions by sampling observed donors with replacement."""
    donors = trial.dropna()
    result = trial.copy()
    sampled = donors.iloc[
        rng.integers(0, len(donors), size=len(held_positions))
    ]
    result.iloc[held_positions] = list(sampled.array)
    return result


def _evaluate_column(frame: pd.DataFrame, eligible: list[str], predictors: list[str], column: str, method: str, neighbours: int, iterations: int, repeats: int, holdout: float, seed: int) -> dict[str, object]:
    observed_positions = np.flatnonzero(frame[column].notna().to_numpy())
    kind = _kind(frame[column])
    if len(observed_positions) < 6:
        return {"Predictor": str(column), "Type": kind.title(), "Metric": "", "Score": np.nan, "Random baseline": np.nan, "Improvement": np.nan, "Status": "Too few observed values"}
    scores, baselines = [], []
    for repeat in range(max(1, int(repeats))):
        rng = np.random.default_rng(seed + repeat * 1009 + sum(map(ord, str(column))))
        count = min(len(observed_positions) - 2, max(2, round(len(observed_positions) * holdout)))
        held = rng.choice(observed_positions, size=count, replace=False)
        trial = frame.copy()
        truth = frame[column].iloc[held].copy()
        trial.iloc[held, trial.columns.get_loc(column)] = pd.NA
        predicted, _ = _impute_frame(trial, eligible, predictors, method, neighbours, iterations, seed + repeat)
        baseline = _random_donor_baseline(trial[column], held, rng)
        metric, value = _score(truth, predicted[column].iloc[held], kind)
        _, base = _score(truth, baseline.iloc[held], kind)
        scores.append(value)
        baselines.append(base)
    score = float(np.mean(scores))
    baseline = float(np.mean(baselines))
    if metric == "Accuracy":
        denominator = 1 - baseline
        improvement = (score - baseline) / denominator if denominator > 0 else 0.0
    else:
        improvement = 1 - score / baseline if baseline > 0 else 0.0
    return {"Predictor": str(column), "Type": kind.title(), "Metric": metric, "Score": score, "Random baseline": baseline, "Improvement": improvement, "Status": "Assessed"}


def _analyse(data: proxy_data, method: str, neighbours: int, iterations: int, repeats: int, holdout: float, seed: int, jobs: int) -> ImputationResult:
    frame = data.frame.copy()
    eligible, excluded = _eligible_columns(frame, data.role_map)
    predictors = _predictor_columns(frame, data.role_map)
    transformer = ImputationStep(
        tuple(eligible), tuple(predictors), method, neighbours, iterations, seed,
    ).fit(frame)
    imputed = transformer.transform(frame)
    methods = transformer.methods_
    assessed = Parallel(n_jobs=max(1, int(jobs)), prefer="threads")(
        delayed(_evaluate_column)(frame, eligible, predictors, column, method, neighbours, iterations, repeats, holdout, seed)
        for column in eligible
    )
    evaluation = pd.DataFrame(assessed)
    if not evaluation.empty:
        evaluation.insert(2, "Missing", [int(frame[column].isna().sum()) for column in eligible])
        evaluation.insert(3, "Method used", [methods.get(column, "") for column in eligible])
        evaluation["Score"] = evaluation["Score"].round(4)
        evaluation["Random baseline"] = evaluation["Random baseline"].round(4)
        evaluation["Improvement"] = evaluation["Improvement"].round(3)
    before = frame.isna().sum()
    after = imputed.isna().sum()
    rows = []
    for column in frame.columns:
        if before[column] == 0:
            continue
        rows.append({
            "Predictor": str(column), "Before": int(before[column]), "After": int(after[column]),
            "Status": excluded.get(column, methods.get(column, "Imputed")),
        })
    return ImputationResult(imputed, pd.DataFrame(rows), evaluation, transformer)


def _apply_analysis(
    source: proxy_data,
    analysis: ImputationResult,
    *,
    step_name: str = "miss_impute",
    operation: str = "Learned Imputation",
) -> proxy_data:
    """Register learned imputation and its full-data display preview."""
    return source.with_pipeline_step(
        analysis.transformer,
        name=step_name,
        operation=operation,
        preview_frame=analysis.frame,
    )


def _performance_table(
    evaluation: pd.DataFrame,
    minimum_improvement: float,
) -> pd.DataFrame:
    """Classify and order imputation assessments for display."""
    table = evaluation.copy()
    if table.empty:
        return table
    assessed = table["Status"].eq("Assessed")
    strong = assessed & table["Improvement"].gt(float(minimum_improvement))
    table.loc[assessed, "Status"] = "Weak"
    table.loc[strong, "Status"] = "Strong"
    return table.sort_values(
        ["Improvement", "Predictor"],
        ascending=[False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def _performance_row_styles(table: pd.DataFrame) -> list[dict[str, object]]:
    """Return semantic row colours for the performance DataTable."""
    if "Status" not in table.columns:
        return []
    styles = []
    for status, class_name in IMPUTATION_ROW_CLASSES.items():
        rows = table.index[table["Status"].eq(status)].tolist()
        if rows:
            styles.append({"rows": rows, "class": class_name})
    return styles


def _missingness_figure(summary: pd.DataFrame, applied: bool, full_screen: bool = False) -> go.Figure:
    if summary.empty:
        return Card.empty_figure("The data contains no missing values")
    ordered = summary.sort_values(["Before", "Predictor"], ascending=[True, True])
    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=ordered["Before"], y=ordered["Predictor"], orientation="h", name="Before",
        marker_color="#9aa7b2", customdata=ordered["Status"],
        hovertemplate="%{y}: %{x:,} missing before<br>%{customdata}<extra></extra>",
    ))
    figure.add_trace(go.Bar(
        x=ordered["After"] if applied else ordered["Before"],
        y=ordered["Predictor"], orientation="h", name="After",
        marker_color="#154c79" if applied else "rgba(21,76,121,0.28)",
        customdata=ordered["Status"],
        hovertemplate="%{y}: %{x:,} missing after<br>%{customdata}<extra></extra>",
    ))
    figure.update_layout(
        template="plotly_white", barmode="group", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#bbd6f8", xaxis_title="Missing observations",
        margin={"l": 20, "r": 25, "t": 10, "b": 90},
        modebar={"orientation": "v"},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.28,
            "yanchor": "top",
        },
        xaxis={"rangemode": "tozero", "fixedrange": not full_screen},
        yaxis={"fixedrange": not full_screen},
    )
    return figure


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = "Learned Imputation"
    this.description = "This card imputes eligible predictor values with scikit-learn and evaluates the selected imputation method using repeated artificial masking."

    def front():
        return ui.TagList(
            ui.span("Missing values before and after imputation", class_="text-primary text-center d-block"),
            shinywidgets.output_widget(id="MissingChart", fill=True, guide=this, title="Imputation result", text="Compares missing counts before and after applying the selected method. Excluded predictors retain their missing values.", position="left"),
        )
    this.front = front

    def back():
        return ui.TagList(
            ui.span("Cross-validated imputation performance", class_="text-primary text-center d-block"),
            ui.output_ui(id="Performance", guide=this, title="Imputation performance", text="Observed values are repeatedly hidden, imputed without seeing their true values, and compared with random draws from that predictor's remaining observed values.", position="left"),
        )
    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(id="Busy"), ui.output_ui(id="Check"),
            ui.input_checkbox_group(
                id="Apply", label="Imputation", choices=["Apply"],
                inline=True, guide=this, title="Apply imputation", position="top",
                text="The imputation is applied to any missing values when the button is checked.",
            ),
            class_="vertically-scrollable-footer",
        )
    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_radio_buttons(
                id="Method", label="Imputation method", choices=METHOD_LABELS, selected="simple",
                guide=this, text="Choose how missing values are estimated. Median or mode uses each predictor by itself; the other methods use relationships between continuous predictors. Categorical predictors are always filled with their most common value.", position="left"
            ),
            ui.input_slider(
                id="Neighbours", label="KNN neighbors", min=2, max=10, value=5, step=1,
                guide=this, text="The number of nearest neighbors to use with KNN imputation. A smaller number makes a more complex model.", position="left"
            ),
            ui.input_slider(
                id="Iterations", label="Iterative-Imputer maximum iterations", min=2, max=30, value=10, step=1,
                guide=this, text="The number of iterations to use with Iterative imputation.", position="left"
            ),
            ui.input_slider(
                id="Repeats", label="Validation repeats", min=1, max=10, value=3, step=1,
                guide=this, text="The number validations repeats to fairly assess all forms of imputation.", position="left"
            ),
            ui.input_slider(
                id="Holdout", label="Observed values hidden per repeat (%)", min=5, max=30, value=15, step=5,
                guide=this, text="The number of values whose imputation is to be assessed against their known value (per repeat).", position="left"
            ),
            ui.input_slider(
                id="MinImprovement", label="Minimum improvement over random baseline", min=0, max=1, value=0.25, step=0.05,
                guide=this, text="An assessed imputer must exceed this proportional improvement to be classified Strong; otherwise it is Weak.", position="left"
            ),
            ui.input_slider(
                id="Jobs", label="Parallel evaluation workers", min=1, max=max(1, min(4, os.cpu_count() or 1)), value=max(1, min(2, os.cpu_count() or 1)), step=1,
                guide=this, text="The number of cpu processes that can run concurrently.", position="left"
            ),
        )
    this.settings = settings

    def server(input, output, session):
        busy = this.busy()

        @this.suspendable(calc=True)
        def incomingproxy_data():
            return this.input_data()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def Options():
            return {
                "method": str(input.Method()), "neighbours": int(input.Neighbours()),
                "iterations": int(input.Iterations()), "repeats": int(input.Repeats()),
                "holdout": float(input.Holdout()) / 100, "seed": 2025,
                "jobs": int(input.Jobs()),
            }

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinImprovement():
            return float(input.MinImprovement())

        @busy.track("Imputing and cross-validating missing values…")
        @reactive.extended_task
        async def Calculate(data: proxy_data, options: dict[str, object]):
            return await asyncio.to_thread(_analyse, data, **options)

        @this.suspendable()
        def StartAnalysis():
            Calculate.invoke(incomingproxy_data().clone(), Options())

        @this.suspendable(calc=True)
        @this.record_code
        def Analysis():
            return Calculate.result()

        @this.suspendable(calc=True)
        @this.record_code
        def TransformedData():
            source = incomingproxy_data()
            if "Apply" not in (input.Apply() or []):
                return source.with_inactive_step(
                    stage="Learning",
                    card=this.namespace,
                    operation=this.long_name,
                    parameters=Options(),
                )
            analysis = Analysis()
            return _apply_analysis(
                source,
                analysis,
                step_name=this.namespace,
                operation=this.long_name,
            )

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render_widget
        def MissingChart():
            full_screen = bool(this.isFullScreen())
            applied = "Apply" in (input.Apply() or [])
            figure = _missingness_figure(Analysis().summary, applied, full_screen)
            widget = go.FigureWidget(figure)
            widget._config = getattr(widget, "_config", {}) | {"displayModeBar": full_screen, "displaylogo": False, "responsive": True}
            return widget

        @output
        @render.ui
        def Performance():
            return ui.output_data_frame(id="PerformanceTable")

        @output
        @render.data_frame
        def PerformanceTable():
            table = _performance_table(
                Analysis().evaluation,
                MinImprovement(),
            )
            return render.DataTable(
                table,
                width="100%",
                height="98%",
                styles=_performance_row_styles(table),
            )

        @output
        @render.ui
        def Check():
            analysis = Analysis()
            eligible = int((analysis.summary["Before"] > analysis.summary["After"]).sum()) if not analysis.summary.empty else 0
            remaining = int(analysis.summary["After"].sum()) if not analysis.summary.empty else 0
            applied = "Apply" in (input.Apply() or [])
            if applied:
                return ui.span(f"Imputed {eligible} predictors; {remaining} missing values remain in excluded predictors.", class_="text-warning" if remaining else "text-success")
            return ui.span(f"Ready to impute {eligible} predictors using {METHOD_LABELS.get(Options()['method'], Options()['method'])}.", class_="text-primary")

        session.on_ended(Calculate.cancel)

        return TransformedData

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
