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
from list_pandas import is_list
from module import Module
from proxy_data import proxy_data
from roles import Role
from shiny import reactive, render, req, ui
from shinywidgets import render_widget

ROW_ORDER = "__row_order__"
REFERENCE_LABELS = {
    "overall": "All observations",
    "first": "First sequence group",
    "previous": "Previous sequence group",
}
CALIBRATION_PERMUTATIONS = 100
CALIBRATION_QUANTILE = 0.95
CALIBRATION_SEED = 2025
SUMMARY_ROW_CLASSES = {
    "Strong": "data-homogeneous-strong-row",
    "Moderate": "data-homogeneous-moderate-row",
    "Weak": "data-homogeneous-weak-row",
    "Stable": "data-homogeneous-stable-row",
    "Excluded": "data-homogeneous-excluded-row",
}


@dataclass
class HomogeneityAnalysis:
    scores: pd.DataFrame
    summary: pd.DataFrame
    excluded: pd.DataFrame
    observations: int
    groups: int
    sequence_label: str


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


def _sequence_candidates(data: proxy_data) -> dict[str, str]:
    """Return role-assigned or unique scalar sequence choices.

    Explicit Sequence and Identifier roles may legitimately contain repeats
    (for example, longitudinal data).  Without those roles, uniqueness is the
    defensible signal that a column can order observations without creating
    arbitrary ties.  Complete unique columns precede unique columns containing
    missing values because missing sequence values must be placed last.
    """
    frame = data.frame
    ordered: list[str] = []
    compatible = {
        column for column in frame.columns
        if _kind(frame[column]) in {
            "numeric", "datetime", "categorical", "ordered", "boolean",
        }
    }
    for role in (Role.SEQUENCE, Role.IDENTIFIER):
        ordered.extend(
            column for column in frame.columns
            if column in compatible
            and column in data.role_map.columns_with_role(role)
            and column not in ordered
        )

    unique = []
    for position, column in enumerate(frame.columns):
        series = frame[column]
        if column in ordered or column not in compatible:
            continue
        observed = int(series.notna().sum())
        cardinality = int(series.nunique(dropna=True))
        if observed >= 2 and cardinality == observed:
            unique.append((series.isna().any(), position, column))
    ordered.extend(column for _, _, column in sorted(unique))
    return {ROW_ORDER: "Current row order"} | {
        str(column): str(column) for column in ordered
    }


def _eligible_columns(
    data: proxy_data,
    *,
    maximum_levels: int = 30,
) -> tuple[list[str], dict[str, str]]:
    """Return analysable columns and explicit reasons for exclusions."""
    frame = data.frame
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    for column in frame.columns:
        name = str(column)
        roles = data.role_map.roles_for(column)
        kind = _kind(frame[column])
        if name.startswith(Card.SHADOW_PREFIX):
            excluded[name] = "Shadow variable"
        elif roles & {Role.SEQUENCE, Role.IDENTIFIER, Role.GEOMETRY}:
            excluded[name] = "Sequence, identifier, or geometry role"
        elif kind in {"cyclic", "list", "geometry", "unsupported"}:
            excluded[name] = f"{kind.title()} variables are not analysed"
        elif frame[column].notna().sum() < 2:
            excluded[name] = "Too few observed values"
        elif kind == "categorical" and frame[column].nunique(dropna=True) > maximum_levels:
            excluded[name] = f"More than {maximum_levels} observed levels"
        else:
            eligible.append(name)
    return eligible, excluded


def _order_frame(frame: pd.DataFrame, sequence: str) -> pd.DataFrame:
    result = frame.copy()
    result["__original_position__"] = np.arange(len(result))
    if sequence != ROW_ORDER and sequence in result.columns:
        try:
            result = result.sort_values(
                [sequence, "__original_position__"],
                kind="stable",
                na_position="last",
            )
        except TypeError:
            keys = result[sequence].astype("string")
            result = result.assign(__sequence_key__=keys).sort_values(
                ["__sequence_key__", "__original_position__"],
                kind="stable",
                na_position="last",
            ).drop(columns="__sequence_key__")
    return result.drop(columns="__original_position__").reset_index(drop=True)


def _assign_groups(length: int, requested: int, minimum_size: int = 5) -> np.ndarray:
    if length == 0:
        return np.array([], dtype=int)
    maximum = max(1, length // max(1, int(minimum_size)))
    count = min(max(1, int(requested)), maximum)
    groups = np.empty(length, dtype=int)
    for group, positions in enumerate(np.array_split(np.arange(length), count), start=1):
        groups[positions] = group
    return groups


def _numeric_values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        values = series.astype("datetime64[ns]").astype("int64").to_numpy(dtype=float)
        values[series.isna().to_numpy()] = np.nan
        return values
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _numeric_drift(sample: pd.Series, reference: pd.Series) -> float:
    left = _numeric_values(sample)
    right = _numeric_values(reference)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return np.nan
    combined = np.concatenate([left, right])
    scale = float(np.nanpercentile(combined, 75) - np.nanpercentile(combined, 25))
    if not np.isfinite(scale) or scale == 0:
        scale = float(np.nanstd(combined))
    if not np.isfinite(scale) or scale == 0:
        return 0.0 if np.array_equal(np.sort(left), np.sort(right)) else 1.0
    # One-dimensional Wasserstein distance calculated from empirical quantiles.
    quantiles = np.linspace(0, 1, max(len(left), len(right)))
    distance = float(np.mean(np.abs(
        np.quantile(left, quantiles) - np.quantile(right, quantiles)
    )))
    return float(1.0 - np.exp(-distance / scale))


def _categorical_drift(sample: pd.Series, reference: pd.Series) -> float:
    # Missingness is measured separately so it cannot inflate both components.
    left = sample.dropna().astype("string")
    right = reference.dropna().astype("string")
    if left.empty or right.empty:
        return np.nan
    levels = left.unique().tolist()
    levels.extend(value for value in right.unique() if value not in levels)
    if not levels:
        return np.nan
    p = left.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(float)
    q = right.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(float)
    midpoint = (p + q) / 2

    def divergence(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log(values[mask] / midpoint[mask])))

    js = (divergence(p) + divergence(q)) / 2
    return float(np.sqrt(max(0.0, js) / np.log(2)))


def _distribution_drift(sample: pd.Series, reference: pd.Series, kind: str) -> float:
    if kind in {"numeric", "datetime"}:
        return _numeric_drift(sample, reference)
    return _categorical_drift(sample, reference)


def _pair_raw_drift(left: pd.Series, right: pd.Series, kind: str) -> tuple[float, float, float]:
    """Return distribution, missingness, and combined raw discrepancies."""
    distribution = _distribution_drift(left, right, kind)
    missingness = abs(float(left.isna().mean()) - float(right.isna().mean()))
    combined = (
        float(np.nanmax([distribution, missingness]))
        if np.isfinite(distribution)
        else missingness
    )
    return distribution, missingness, combined


def _raw_group_scores(
    series: pd.Series,
    groups: np.ndarray,
    kind: str,
    reference: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate uncalibrated discrepancies for all consecutive groups."""
    distribution, missingness, combined = [], [], []
    for group in sorted(np.unique(groups)):
        sample = series.loc[groups == group]
        if reference == "first":
            comparison = series.loc[groups == 1]
        elif reference == "previous":
            comparison = series.loc[groups == max(1, group - 1)]
        else:
            comparison = series
        values = _pair_raw_drift(sample, comparison, kind)
        distribution.append(values[0])
        missingness.append(values[1])
        combined.append(values[2])
    return (
        np.asarray(distribution, dtype=float),
        np.asarray(missingness, dtype=float),
        np.asarray(combined, dtype=float),
    )


def _chance_correct(raw, boundary: float, random_scale: float | None = None):
    """Scale excess beyond a random-order boundary by random variability."""
    values = np.asarray(raw, dtype=float)
    if not np.isfinite(boundary):
        corrected = np.full(values.shape, np.nan)
    else:
        scale = float(random_scale) if random_scale is not None else np.nan
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            scale = max(1.0 - boundary, np.finfo(float).eps)
        corrected = np.clip((values - boundary) / scale, 0.0, 1.0)
    if np.ndim(raw) == 0:
        return float(corrected)
    return corrected


def _series_summary(series: pd.Series, kind: str) -> str:
    missing = float(series.isna().mean()) if len(series) else np.nan
    observed = series.dropna()
    if observed.empty:
        detail = "no observed values"
    elif kind == "numeric":
        detail = f"median {float(observed.median()):.4g}"
    elif kind == "datetime":
        detail = f"median {pd.to_datetime(observed.astype('int64').median(), unit='ns').date()}"
    else:
        mode = observed.mode(dropna=True)
        label = mode.iloc[0] if not mode.empty else observed.iloc[0]
        detail = f"most common {label!s} ({float((observed == label).mean()):.0%})"
    return f"{detail}; {missing:.0%} missing"


def _sequence_interval(frame: pd.DataFrame, positions: np.ndarray, sequence: str) -> str:
    if sequence == ROW_ORDER or sequence not in frame.columns:
        return f"Rows {int(positions[0]) + 1}–{int(positions[-1]) + 1}"
    values = frame.loc[positions, sequence].dropna()
    if values.empty:
        return "Missing sequence values"
    return f"{values.iloc[0]} – {values.iloc[-1]}"


def _reference_series(
    frame: pd.DataFrame,
    groups: np.ndarray,
    group: int,
    column: str,
    reference: str,
) -> pd.Series:
    if reference == "first":
        return frame.loc[groups == 1, column]
    if reference == "previous":
        target = max(1, group - 1)
        return frame.loc[groups == target, column]
    return frame[column]


def _trend_label(frame: pd.DataFrame, groups: np.ndarray, column: str, kind: str) -> str:
    if kind not in {"numeric", "datetime"}:
        return "Not applicable"
    values = _numeric_values(frame[column])
    medians = []
    for group in sorted(np.unique(groups)):
        group_values = values[groups == group]
        observed = group_values[np.isfinite(group_values)]
        medians.append(float(np.median(observed)) if len(observed) else np.nan)
    medians = np.asarray(medians, dtype=float)
    valid = np.isfinite(medians)
    if valid.sum() < 3 or np.nanstd(medians[valid]) == 0:
        return "No clear trend"
    correlation = float(np.corrcoef(np.arange(len(medians))[valid], medians[valid])[0, 1])
    if abs(correlation) < 0.5:
        return "No clear trend"
    return f"{'Increasing' if correlation > 0 else 'Decreasing'} ({abs(correlation):.2f})"


def _status(score: float, threshold: float) -> str:
    if not np.isfinite(score):
        return "Excluded"
    if score >= threshold:
        return "Strong"
    if score >= threshold / 2:
        return "Moderate"
    if score > 0.02:
        return "Weak"
    return "Stable"


def _analyse_homogeneity(
    data: proxy_data,
    *,
    sequence: str,
    variables: list[str],
    group_count: int,
    reference: str,
    threshold: float,
    maximum_levels: int = 30,
    permutations: int = CALIBRATION_PERMUTATIONS,
    random_state: int = CALIBRATION_SEED,
) -> HomogeneityAnalysis:
    frame = _order_frame(data.frame, sequence)
    eligible, exclusions = _eligible_columns(data, maximum_levels=maximum_levels)
    selected = [column for column in variables if column in eligible]
    groups = _assign_groups(len(frame), group_count)
    group_values = sorted(np.unique(groups))
    rng = np.random.default_rng(random_state)
    permutation_seeds = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=max(1, int(permutations)),
        dtype=np.uint32,
    )
    rows: list[dict[str, object]] = []
    calibration: dict[str, dict[str, float]] = {}
    for column in selected:
        kind = _kind(frame[column])
        series = frame[column].reset_index(drop=True)
        distribution, missingness, raw = _raw_group_scores(
            series, groups, kind, reference
        )
        null_maxima, null_first_last = [], []
        for seed in permutation_seeds:
            order = np.random.default_rng(int(seed)).permutation(len(frame))
            shuffled = series.iloc[order].reset_index(drop=True)
            _, _, null_raw = _raw_group_scores(
                shuffled, groups, kind, reference
            )
            null_maxima.append(float(np.nanmax(null_raw)))
            first = shuffled.loc[groups == group_values[0]]
            last = shuffled.loc[groups == group_values[-1]]
            null_first_last.append(_pair_raw_drift(first, last, kind)[2])
        boundary = float(np.nanquantile(null_maxima, CALIBRATION_QUANTILE))
        random_scale = boundary - float(np.nanmedian(null_maxima))
        first_last_boundary = float(
            np.nanquantile(null_first_last, CALIBRATION_QUANTILE)
        )
        first_last_scale = first_last_boundary - float(
            np.nanmedian(null_first_last)
        )
        corrected = _chance_correct(raw, boundary, random_scale)
        calibration[column] = {
            "boundary": boundary,
            "random_scale": random_scale,
            "first_last_boundary": first_last_boundary,
            "first_last_scale": first_last_scale,
        }
        for position, group in enumerate(group_values):
            positions = np.flatnonzero(groups == group)
            sample = frame.loc[positions, column]
            comparison = _reference_series(frame, groups, group, column, reference)
            rows.append({
                "Group": int(group),
                "Sequence interval": _sequence_interval(frame, positions, sequence),
                "Variable": column,
                "Type": kind.title(),
                "Distribution drift": distribution[position],
                "Missingness drift": missingness[position],
                "Raw drift": raw[position],
                "Random boundary": boundary,
                "Drift": corrected[position],
                "Group summary": _series_summary(sample, kind),
                "Reference summary": _series_summary(comparison, kind),
            })
    scores = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for column in selected:
        part = scores.loc[scores["Variable"] == column]
        maximum = float(part["Drift"].max())
        peak = part.loc[part["Raw drift"].idxmax(), "Sequence interval"]
        first = frame.loc[groups == groups.min(), column]
        last = frame.loc[groups == groups.max(), column]
        kind = _kind(frame[column])
        first_last_raw = _pair_raw_drift(first, last, kind)[2]
        first_last = _chance_correct(
            first_last_raw,
            calibration[column]["first_last_boundary"],
            calibration[column]["first_last_scale"],
        )
        summaries.append({
            "Variable": column,
            "Type": kind.title(),
            "Overall drift": float(part["Drift"].mean()),
            "Maximum drift": maximum,
            "Raw maximum": float(part["Raw drift"].max()),
            "Random boundary": calibration[column]["boundary"],
            "Peak location": peak,
            "First-to-last drift": first_last,
            "Trend": _trend_label(frame, groups, column, kind),
            "Maximum missingness difference": float(part["Missingness drift"].max()),
            "Status": _status(maximum, threshold),
            "Reason": "",
        })
    excluded = pd.DataFrame([
        {"Variable": column, "Reason": reason}
        for column, reason in exclusions.items()
    ])
    for row in excluded.to_dict("records"):
        if row["Variable"] != sequence:
            summaries.append({
                "Variable": row["Variable"], "Type": "Excluded",
                "Overall drift": np.nan, "Maximum drift": np.nan,
                "Raw maximum": np.nan, "Random boundary": np.nan,
                "Peak location": "", "First-to-last drift": np.nan,
                "Trend": "Not applicable", "Maximum missingness difference": np.nan,
                "Status": "Excluded", "Reason": row["Reason"],
            })
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        rank = {"Strong": 0, "Moderate": 1, "Weak": 2, "Stable": 3, "Excluded": 4}
        summary["__rank"] = summary["Status"].map(rank).fillna(5)
        summary = summary.sort_values(
            ["__rank", "Maximum drift", "Variable"],
            ascending=[True, False, True], na_position="last", kind="stable",
        ).drop(columns="__rank").reset_index(drop=True)
        for column in (
            "Overall drift", "Maximum drift", "Raw maximum", "Random boundary",
            "First-to-last drift", "Maximum missingness difference",
        ):
            summary[column] = summary[column].round(3)
    label = "Current row order" if sequence == ROW_ORDER else str(sequence)
    return HomogeneityAnalysis(
        scores=scores,
        summary=summary,
        excluded=excluded,
        observations=len(frame),
        groups=int(groups.max()) if len(groups) else 0,
        sequence_label=label,
    )


def _homogeneity_figure(
    analysis: HomogeneityAnalysis,
    *,
    threshold: float,
    full_screen: bool = False,
) -> go.Figure:
    scores = analysis.scores
    if scores.empty:
        return Card.empty_figure("No supported variables are selected")
    variables = scores["Variable"].drop_duplicates().tolist()
    groups = sorted(scores["Group"].unique())
    indexed = scores.set_index(["Group", "Variable"])
    z = np.array([
        [indexed.loc[(group, variable), "Drift"] for group in groups]
        for variable in variables
    ])
    hover = np.array([[
        (
            f"<b>{variable}</b><br>{indexed.loc[(group, variable), 'Sequence interval']}"
            f"<br>Excess drift: {indexed.loc[(group, variable), 'Drift']:.3f}"
            f"<br>Raw drift: {indexed.loc[(group, variable), 'Raw drift']:.3f}"
            f"<br>Random maximum boundary: {indexed.loc[(group, variable), 'Random boundary']:.3f}"
            f"<br>Raw distribution drift: {indexed.loc[(group, variable), 'Distribution drift']:.3f}"
            f"<br>Raw missingness difference: {indexed.loc[(group, variable), 'Missingness drift']:.3f}"
            f"<br>Group: {indexed.loc[(group, variable), 'Group summary']}"
            f"<br>Reference: {indexed.loc[(group, variable), 'Reference summary']}"
        )
        for group in groups
    ] for variable in variables], dtype=object)
    figure = go.Figure(go.Heatmap(
        z=z,
        x=groups,
        y=variables,
        zmin=0,
        zmax=1,
        colorscale=[
            [0.0, "#f7fbff"],
            [float(threshold) / 2, "#c6dbef"],
            [float(threshold), "#6baed6"],
            [(float(threshold) + 1) / 2, "#2171b5"],
            [1.0, "#7b3f00"],
        ],
        colorbar={"title": "Excess<br>drift", "thickness": 14},
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 100, "r": 55, "t": 10, "b": 60},
        xaxis={
            "title": f"Consecutive groups ordered by {analysis.sequence_label}",
            "dtick": 1, "fixedrange": not full_screen,
        },
        yaxis={
            "title": "Variables", "autorange": "reversed",
            "fixedrange": not full_screen,
        },
    )
    return figure


def _summary_row_styles(table: pd.DataFrame) -> list[dict[str, object]]:
    if "Status" not in table.columns:
        return []
    styles = []
    for status, class_name in SUMMARY_ROW_CLASSES.items():
        rows = table.index[table["Status"].eq(status)].tolist()
        if rows:
            styles.append({"rows": rows, "class": class_name})
    return styles


def instance():
    this = Card(file=__file__, mutable=False)
    this.long_name = "Data homogeneity"
    this.description = "This card examines whether variable distributions remain homogeneous through a meaningful observation sequence; a time sequence makes this a data-drift analysis."

    def front():
        return ui.TagList(
            ui.span("Distribution drift through the observation sequence", class_="text-primary text-center d-block"),
            shinywidgets.output_widget(
                id="DriftChart", fill=True, guide=this,
                title="Data homogeneity heatmap",
                text="Each cell measures drift beyond the variable's 95% random-order boundary. Hover details retain the raw distribution and missingness discrepancies.",
                position="left",
            ),
        )
    this.front = front

    def back():
        return ui.TagList(
            ui.span("Variables ranked by evidence of drift", class_="text-primary text-center d-block"),
            ui.output_ui(
                id="Summary", guide=this, title="Homogeneity summary",
                text="The table ranks variables by maximum drift and identifies where the strongest change occurs.", position="left",
            ),
        )
    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(id="Busy"),
            ui.output_ui(id="Check"),
            class_="vertically-scrollable-footer text-center",
        )
    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_select(
                id="Sequence", label="Observation sequence",
                choices={ROW_ORDER: "Current row order"}, selected=ROW_ORDER,
                guide=this, title="Observation sequence", position="left", text="""
                Determines how rows are sorted before grouping. Try more than one plausible sequence where appropriate. Lexicographically sorted string identifiers are meaningful only if their text order represents the intended order—for example, `A1, A10, A2` is not numeric order.
                Sequence-role variables are preferred, followed by identifiers and scalar variables whose observed values are 100% unique. Complete unique variables precede those containing missing values. <em>Current row order</em> is always available."
                """),
            ui.input_selectize(
                id="Variables", label="Variables to examine", choices=[],
                selected=[], multiple=True, options={"plugins": ["remove_button"]},
                guide=this, title="Variables to examine", position="left", text="""
                Controls the heat-map rows. Reducing the selection can make patterns easier to see. Sequence, identifier, geometry, shadow, list, cyclic, unsupported, nearly empty, and high-cardinality categorical variables can be omitted from the analysis. Exclusion reasons appear on the flip-side.
                Structured and high-cardinality variables that cannot be compared defensibly are excluded.
                """,
            ),
            ui.input_slider(
                id="Groups", label="Maximum consecutive groups", min=2, max=50, value=20, step=1,
                guide=this, position="left", text="""
                The largest number of consecutive groups controls resolution. More groups provide greater localization but fewer observations per comparison and therefore more variability. Fewer groups produce more stable distribution estimates but can conceal short-lived changes. The card enforces a minimum of five observations per group, although larger groups are preferable for reliable distribution comparisons.
                The number of groups reduced automatically so every group has at least five observations.
                Changing the number of groups automatically recalibrates every variable. A raw score from one group setting should not be compared directly with a raw score from another; compare their excess-drift results instead."""
            ),
            ui.input_radio_buttons(
                id="Reference", label="Reference distribution", choices=REFERENCE_LABELS, selected="overall",
                guide=this, position="left", text="""Compare each group with all observations (the whole dataset), the first group (the beginning), or the immediately preceding group (a running analysis)."""
            ),
            ui.input_slider(
                id="Threshold", label="Strong drift threshold", min=0.05, max=0.75, value=0.25, step=0.05,
                guide=this, position="left", text="""A variable whose maximum chance-corrected drift reaches this threshold is classified as Strong.
                This acts as a practical classification of variables. It should reflect how much change matters in the dataset rather than being treated as a universal statistical cutoff."""
            ),
        )
    this.settings = settings

    def server(input, output, session):
        busy = this.busy()

        @this.suspendable(calc=True)
        def incomingproxy_data():
            return this.input_data()

        @reactive.effect
        def UpdateSequenceChoices():
            choices = _sequence_candidates(incomingproxy_data())
            keys = list(choices)
            with reactive.isolate():
                previous = input.Sequence()
            selected = previous if previous in choices else (keys[1] if len(keys) > 1 else ROW_ORDER)
            if selected == ROW_ORDER and len(keys) > 1:
                selected = keys[1]
            ui.update_select("Sequence", choices=choices, selected=selected)

        @reactive.effect
        def UpdateVariableChoices():
            choices, _ = _eligible_columns(incomingproxy_data())
            with reactive.isolate():
                previous = list(input.Variables() or [])
            selected = [value for value in previous if value in choices]
            if not selected:
                selected = choices[:12]
            ui.update_selectize("Variables", choices=choices, selected=selected)

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def Options():
            req(input.Sequence() is not None)
            return {
                "sequence": str(input.Sequence()),
                "variables": list(input.Variables() or []),
                "group_count": int(input.Groups()),
                "reference": str(input.Reference()),
                "threshold": float(input.Threshold()),
            }

        @busy.track("Comparing distributions through the observation sequence…")
        @reactive.extended_task
        async def Calculate(data: proxy_data, options: dict[str, object]):
            return await asyncio.to_thread(_analyse_homogeneity, data, **options)

        @this.suspendable()
        def StartAnalysis():
            Calculate.invoke(incomingproxy_data().clone(), Options())

        @this.suspendable(calc=True)
        @this.record_code
        def Analysis():
            return Calculate.result()

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render_widget
        def DriftChart():
            full_screen = bool(this.isFullScreen())
            figure = _homogeneity_figure(
                Analysis(), threshold=float(Options()["threshold"]), full_screen=full_screen,
            )
            figure.update_layout(modebar={"orientation": "v"})
            widget = go.FigureWidget(figure)
            widget._config = getattr(widget, "_config", {}) | {
                "displayModeBar": full_screen, "displaylogo": False, "responsive": True,
            }
            return widget

        @output
        @render.ui
        def Summary():
            return ui.output_data_frame(id="SummaryTable")

        @output
        @render.data_frame
        def SummaryTable():
            table = Analysis().summary
            return render.DataTable(
                table, width="100%", height="98%", styles=_summary_row_styles(table),
            )

        @output
        @render.ui
        def Check():
            analysis = Analysis()
            strong = int(analysis.summary["Status"].eq("Strong").sum()) if not analysis.summary.empty else 0
            analysed = int(analysis.summary["Status"].ne("Excluded").sum()) if not analysis.summary.empty else 0
            excluded = int(analysis.summary["Status"].eq("Excluded").sum()) if not analysis.summary.empty else 0
            excluded_text = f"; {excluded} unsupported variables are listed on the flip-side" if excluded else ""
            return ui.span(
                f"Compared {analysed} variables across {analysis.groups} groups ordered by {analysis.sequence_label}; {strong} show strong excess drift{excluded_text}.",
                class_="text-warning" if strong else "text-success",
            )

        session.on_ended(Calculate.cancel)

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
