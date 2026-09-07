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
from shiny import reactive, render, req, ui
from shinywidgets import render_widget
from text_pandas import text_evidence
from var_types import var_kind

DEFAULT_LOW_CARDINALITY = 15
DEFAULT_HIGH_CARDINALITY = 55
SAMPLE_SEED = 1729
STATUS_ORDER = {
    "Strong review": 0,
    "Review": 1,
    "Expected": 2,
    "Not assessed": 3,
}
STATUS_COLOURS = {
    "Strong review": "#d95f02",
    "Review": "#e6ab02",
    "Expected": "#3978a8",
    "Not assessed": "#9aa7b2",
}
STATUS_ROW_CLASSES = {
    "Strong review": "var-cardinality-strong-review-row",
    "Review": "var-cardinality-review-row",
    "Expected": "var-cardinality-expected-row",
    "Not assessed": "var-cardinality-not-assessed-row",
}


@dataclass(frozen=True)
class CardinalityAnalysis:
    profiles: pd.DataFrame
    source_observations: int
    sampled_observations: int
    sampled: bool


def _freeze(value):
    """Return a stable, hashable representation of common structured values."""
    if isinstance(value, np.ndarray):
        return tuple(_freeze(item) for item in value.tolist())
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _observed(series: pd.Series) -> pd.Series:
    """Drop scalar missing values without interpreting containers as missing."""
    try:
        return series.loc[series.notna()]
    except (TypeError, ValueError):
        keep = []
        for value in series:
            try:
                missing = pd.isna(value)
                keep.append(not isinstance(missing, (bool, np.bool_)) or not missing)
            except (TypeError, ValueError):
                keep.append(True)
        return series.loc[keep]


def _frozen_values(series: pd.Series) -> pd.Series:
    return _observed(series).map(_freeze)


def _bounded_cardinality(series: pd.Series, limit: int) -> tuple[int, bool]:
    """Count distinct values until ``limit`` is exceeded.

    The Boolean result indicates whether the returned count is exact. This
    gives exact low-cardinality results while allowing high-cardinality
    columns to terminate quickly and with bounded memory.
    """
    distinct = set()
    for value in _observed(series):
        distinct.add(_freeze(value))
        if len(distinct) > limit:
            return len(distinct), False
    return len(distinct), True


def _sample_positions(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=int)
    return np.sort(
        np.random.default_rng(SAMPLE_SEED).choice(
            length, size=maximum, replace=False,
        )
    )


def _safe_frequency_summary(series: pd.Series) -> tuple[float, int]:
    values = _frozen_values(series)
    if values.empty:
        return np.nan, 0
    counts = values.value_counts(dropna=False)
    return float(counts.iloc[0] / len(values)), int(counts.eq(1).sum())


def _abbreviate(value: object, limit: int = 25) -> str:
    """Keep chart hover text compact; the flip-side retains the full text."""
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _role_label(data: proxy_data, column: object) -> str:
    """Return every current role in a stable, user-facing form."""
    roles = data.role_map.roles_for(column)
    if not roles:
        return "Unassigned"
    return ", ".join(
        sorted(role.value.replace("_", " ").title() for role in roles)
    )


def _assessment(
    *,
    kind: str,
    cardinality: int,
    cardinality_exact: bool,
    uniqueness: float,
    exact_unique: bool,
    role: Role,
    low_threshold: int,
    high_threshold: int,
) -> tuple[str, str, str]:
    """
    Interpret dtype/cardinality combinations before roles are assigned.
    """
    if cardinality == 0:
        return (
            "Not assessed",
            "No non-missing values are available.",
            "Drop this variable.",
        )
    if cardinality_exact and cardinality == 1:
        return (
            "Strong review",
            "The variable is constant across all observed values.",
            "Drop this variable.",
        )
    if kind in {"geometry", "unknown", "complex", "object", "basket"}:
        return (
            "Not assessed",
            f"Ordinary scalar cardinality is not a reliable interpretation for {kind} values.",
            "Use the variable's specialized data type and intended purpose when assigning its role.",
        )

    if kind == "decimal":
        if cardinality <= low_threshold:
            return (
                "Strong review",
                f"It is odd for this decimal variable to contain only {cardinality} distinct observed values.",
                "Check how the variable was measured and curated.",
            )
        elif cardinality <= high_threshold:
            return (
                "Review",
                f"It is curious for this decimal variable to contain only {cardinality} distinct observed values.",
                "Be suspicious of what the variable represents.",
            )
        elif exact_unique:
            return(
                "Expected",
                "It is plausible that every observed decimal value is unique.",
                "",
            )
        else:
            return(
                "Expected",
                "It is normal for this decimal variable to have some duplicated values.",
                "",
            )

    if kind == "text":
        if cardinality <= low_threshold:
            return (
                "Strong review",
                f"It is odd for this text variable to contain only {cardinality} distinct observed values.",
                "This cardinality better suits a nominal data type.",
            )
        elif cardinality <= high_threshold:
            return (
                "Review",
                f"It is curious for this text variable to contain only {cardinality} distinct observed values.",
                "Be suspicious of what the variable represents.",
            )
        elif exact_unique:
            return(
                "Expected",
                "It is normal that every observed text value is unique.",
                "",
            )
        else:
            return(
                "Expected",
                "It is plausible for this text variable to have some duplicated values.",
                "",
            )

    if kind == "integer":
        if cardinality <= low_threshold:
            return (
                "Strong review",
                f"It is only normal for a count-based integer variable to contain only {cardinality} distinct observed values.",
                "If the variable is not a count check whether it is an ordered data type.",
            )
        elif cardinality <= high_threshold:
            return (
                "Review",
                f"It is only normal for a count-based integer variable to contain only {cardinality} distinct observed values.",
                "Verify that the variable is a count.",
            )
        elif exact_unique:
            return(
                "Review",
                "It is curious that every observed decimal value is unique.",
                "Verify that the variable is not an observation identifier.",
            )
        else:
            return(
                "Expected",
                "It is normal for this integer variable to have some duplicated values.",
                "",
            )

    if kind == "nominal":
        if cardinality <= low_threshold:
            return (
                "Expected",
                f"It is normal for a nominal variable to contain only {cardinality} distinct observed values.",
                "",
            )
        elif cardinality <= high_threshold:
            return (
                "Review",
                f"It is ununsual for a nominal variable to contain {cardinality} distinct observed values.",
                "This is only plausible with high observation counts.",
            )
        elif exact_unique:
            if role != "Identifier":
                return(
                    "Strong review",
                    f"It is pointless as a nominal {role} when every observed value is unique.",
                    "Verify that the variable is not an observation identifier.",
                )
            else:
                return(
                    "Expected",
                    "It is expected that every observed value is unique.",
                    "",
                )
        else:
            return(
                "Strong review",
                f"It is weird for a nominal variable to contain {cardinality} distinct observed values.",
                "Verify whether many levels deserve to be merged.",
            )

    if kind == "ordered":
        if cardinality <= low_threshold:
            return (
                "Expected",
                f"It is normal for an ordered variable to contain only {cardinality} distinct observed values.",
                "",
            )
        elif cardinality <= high_threshold:
            return (
                "Strong review",
                f"It is ununsual for an ordered variable to contain {cardinality} distinct observed values.",
                "Verify whether many levels deserve to be merged.",
            )
        elif exact_unique:
            if role != "Identifier":
                return(
                    "Strong review",
                    f"It is pointless as a ordered {role} when every observed value is unique.",
                    "Verify that the variable is not an observation identifier.",
                )
            else:
                return(
                    "Expected",
                    "It is expected that every observed value is unique.",
                    "",
                )
        else:
            return (
                "Strong review",
                f"It is flawed for an ordered variable to contain {cardinality} distinct observed values.",
                "Whatever it is, it is not an ordered variable.",
            )


    if kind == "cyclic":
        if cardinality <= low_threshold:
            return (
                "Expected",
                f"It is normal for a cyclic ordered variable to contain only {cardinality} distinct observed values.",
                "",
            )
        else:
            return (
                "Strong review",
                f"It is only normal for an angle-based cyclic variable to contain {cardinality} distinct observed values.",
                "Verify whether angles are involved.",
            )

    if kind == "logical":
        return (
            "Review",
            "It is likely that this variable is better as an integer variable.",
            "Convert to something else.",
        )

    if kind == "code":
        if cardinality <= low_threshold:
            return (
                "Review",
                "It is likely that this variable is better as a nominal or ordered variable.",
                "Convert to something else.",
            )
        elif cardinality <= high_threshold:
            return (
                "Expected",
                f"It is normal for a code variable to contain {cardinality} distinct observed values.",
                "",
            )
        elif exact_unique:
            if role != "Identifier":
                return(
                    "Strong review",
                    f"It is pointless as a code {role} when every observed value is unique.",
                    "Verify that the variable is not an observation identifier.",
                )
            else:
                return(
                    "Expected",
                    "It is expected that every observed value is unique.",
                    "",
                )
        else:
            return (
                "Expected",
                f"It is normal for a code variable to contain {cardinality} distinct observed values.",
                "",
            )
        return (
            "Not assessed",
            "",
            "",
        )



def _profile_column(
    name: str,
    full: pd.Series,
    sample: pd.Series,
    *,
    role: str,
    sample_was_used: bool,
    low_threshold: int,
    high_threshold: int,
) -> dict[str, object]:
    kind = var_kind(full.dtype)
    total = len(full)
    observed_total = len(_observed(full))
    missing = total - observed_total
    sampled_observed = _observed(sample)
    sampled_values = _frozen_values(sample)
    sampled_cardinality = int(sampled_values.nunique(dropna=False))

    bounded, bounded_exact = _bounded_cardinality(full, high_threshold)
    if bounded_exact:
        cardinality = bounded
        cardinality_exact = True
        basis = "Full data"
    else:
        cardinality = max(bounded, sampled_cardinality)
        cardinality_exact = not sample_was_used
        basis = "Full data" if cardinality_exact else "Sample/lower bound"

    sample_uniqueness = (
        sampled_cardinality / len(sampled_observed)
        if len(sampled_observed) else np.nan
    )
    exact_unique = False
    if observed_total and sample_uniqueness >= 0.95 and kind not in {
        "geometry", "structured", "complex", "unsupported",
    }:
        try:
            exact_unique = bool(_frozen_values(full).is_unique)
        except (TypeError, ValueError):
            exact_unique = False

    dominant_ratio, singletons = _safe_frequency_summary(sample)
    # singleton_ratio = singletons / sampled_cardinality if sampled_cardinality else np.nan
    unused_categories = 0
    if isinstance(full.dtype, pd.CategoricalDtype):
        # Sampling must not turn levels used elsewhere in the complete column
        # into apparent unused levels.
        observed_codes = full.cat.codes.to_numpy()
        used_categories = int(np.unique(observed_codes[observed_codes >= 0]).size)
        unused_categories = max(0, len(full.cat.categories) - used_categories)

    free_text = kind == "text"
    if kind in {"nominal", "categorical", "ordered categorical"}:
        try:
            free_text = text_evidence(
                sample,
                high_cardinality=high_threshold,
                limit=min(2_000, max(1, len(sample))),
            ).matches
        except (TypeError, ValueError):
            free_text = False

    status, explanation, action = _assessment(
        kind=kind,
        cardinality=cardinality,
        cardinality_exact=cardinality_exact,
        uniqueness=float(sample_uniqueness),
        exact_unique=exact_unique,
        role=role,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    distinct_display = f"{cardinality:,}" if cardinality_exact else f"≥{cardinality:,}"
    return {
        "Variable": str(name),
        "Role": role,
        "Raw type": str(full.dtype),
        "Semantic type": "free text" if free_text else kind,
        "Rows analyzed": len(sample),
        "Observed": observed_total,
        "Missing": missing,
        "Distinct": distinct_display,
        "Distinct value": cardinality,
        "Basis": basis,
        "Unique %": round(100 * sample_uniqueness, 1) if np.isfinite(sample_uniqueness) else np.nan,
        "Most frequent %": round(100 * dominant_ratio, 1) if np.isfinite(dominant_ratio) else np.nan,
        "Singleton levels": singletons,
        "Unused levels": unused_categories,
        "Finding": status,
        "Explanation": explanation,
        "Role consideration": action,
    }


def _analyse_cardinality(
    data: proxy_data,
    *,
    maximum_observations: int,
    low_threshold: int,
    high_threshold: int,
) -> CardinalityAnalysis:
    frame = data.frame
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Variable cardinality requires tabular data")
    if len(frame) <= maximum_observations:
        sample = frame
    else:
        sample = frame.iloc[_sample_positions(len(frame), maximum_observations)]
    rows = [
        _profile_column(
            str(column),
            frame[column],
            sample[column],
            role=_role_label(data, column),
            sample_was_used=len(frame) > len(sample),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        for column in frame.columns
        if not str(column).startswith(Card.SHADOW_PREFIX)
    ]
    profiles = pd.DataFrame(rows)
    return CardinalityAnalysis(
        profiles=profiles,
        source_observations=len(frame),
        sampled_observations=len(sample),
        sampled=len(frame) > len(sample),
    )


def _ordered_profiles(profiles: pd.DataFrame, ordering: str) -> pd.DataFrame:
    if profiles.empty:
        return profiles.copy()
    table = profiles.copy()
    table["__position__"] = np.arange(len(table))
    table["__severity__"] = table["Finding"].map(STATUS_ORDER).fillna(99)
    if ordering == "finding":
        table = table.sort_values(
            ["__severity__", "Distinct value", "Variable"],
            ascending=[True, False, True], kind="stable",
        )
    elif ordering == "original":
        table = table.sort_values("__position__", kind="stable")
    else:
        table = table.sort_values(
            ["Distinct value", "Variable"], ascending=[False, True], kind="stable",
        )
    return table.drop(columns=["__position__", "__severity__"]).reset_index(drop=True)


def _profile_row_styles(table: pd.DataFrame) -> list[dict[str, object]]:
    """Color profile rows with the same finding palette as the chart."""
    if "Finding" not in table.columns:
        return []
    styles: list[dict[str, object]] = []
    for finding, class_name in STATUS_ROW_CLASSES.items():
        rows = table.index[table["Finding"].eq(finding)].tolist()
        if rows:
            styles.append({"rows": rows, "class": class_name})
    return styles


def _cardinality_figure(
    analysis: CardinalityAnalysis,
    *,
    ordering: str,
    logarithmic: bool,
    show_thresholds: bool,
    low_threshold: int,
    high_threshold: int,
    full_screen: bool,
) -> go.Figure:
    if analysis.profiles.empty:
        return Card.empty_figure("No original variables are available")
    table = _ordered_profiles(analysis.profiles, ordering)
    custom = np.column_stack([
        table["Distinct"],
        table["Unique %"].map(lambda value: "—" if pd.isna(value) else f"{value:.1f}%"),
        table["Observed"].map(lambda value: f"{value:,}"),
        table["Missing"].map(lambda value: f"{value:,}"),
        table["Role"],
        table["Raw type"],
        table["Semantic type"],
        table["Finding"],
        table["Explanation"].map(_abbreviate),
        table["Basis"],
    ])
    figure = go.Figure(go.Bar(
        x=table["Distinct value"],
        y=table["Variable"],
        orientation="h",
        marker_color=table["Finding"].map(STATUS_COLOURS),
        customdata=custom,
        hovertemplate=(
            "<b>%{y}</b><br>Distinct: %{customdata[0]}"
            "<br>Uniqueness in analysis: %{customdata[1]}"
            "<br>Observed in full data: %{customdata[2]}"
            "<br>Missing in full data: %{customdata[3]}"
            "<br>Role: %{customdata[4]}"
            "<br>Raw type: %{customdata[5]}"
            "<br>Semantic type: %{customdata[6]}"
            "<br>Finding: %{customdata[7]}"
            "<br>%{customdata[8]}"
            "<br>Basis: %{customdata[9]}<extra></extra>"
        ),
    ))
    if show_thresholds:
        for threshold, label in (
            (low_threshold, "Low-cardinality limit"),
            (high_threshold, "High-cardinality limit"),
        ):
            # Plotly shape coordinates on logarithmic axes are base-10
            # exponents, unlike the data supplied to the bar trace.
            # position = float(np.log10(threshold)) if logarithmic else threshold
            position = threshold
            figure.add_vline(
                x=position,
                line_dash="dot",
                line_color="#343a40",
                annotation_text=label if full_screen else None,
                annotation_position="top",
            )
    sample_note = (
        f"deterministic sample of {analysis.sampled_observations:,} from "
        f"{analysis.source_observations:,} rows"
        if analysis.sampled
        else f"all {analysis.source_observations:,} rows"
    )
    axis_range = None
    if logarithmic:
        upper = float(np.ceil(np.log10(max(1, analysis.source_observations))))
        # Keep a usable interval for the one-row case while retaining 10^0 as
        # the requested upper bound.
        axis_range = [min(0.0, upper - 1.0), upper]
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#bbd6f8",
        margin={"l": 120 if full_screen else 85, "r": 35, "t": 30, "b": 65},
        xaxis={
            "title": f"Distinct observed values — {sample_note}",
            "type": "log" if logarithmic else "linear",
            "range": axis_range,
            "fixedrange": not full_screen,
        },
        yaxis={"autorange": "reversed", "fixedrange": not full_screen},
        showlegend=False,  # shows an unwanted entry for "trace 0"
        modebar={"orientation": "v"},
        modebar_remove=[
            "select2d", "lasso2d", "toggleHover", "toggleSpikelines",
            "hoverClosestCartesian", "hoverCompareCartesian",
        ]
    )

    for status, colour in STATUS_COLOURS.items():
        figure.add_trace(go.Bar(
            x=[None], y=[None], name=status, marker_color=colour,
            hoverinfo="skip", showlegend=True,
        ))
    figure.update_layout(
        legend={
            "orientation": "h", "x": 0.5, "xanchor": "center",
            "y": -0.22, "yanchor": "top",
        }
    )
    return figure


def instance():
    """Create the immutable variable-cardinality card."""
    this = Card(file=__file__, mutable=False)
    this.long_name = "Variable cardinality"
    this.description = "This card profiles distinct values and highlights data-type and cardinality combinations to review."

    def front():
        return ui.TagList(
            ui.span(
                "Variable cardinality chart",
                class_="text-primary text-center d-block",
            ),
            shinywidgets.output_widget(
                id="Chart",
                fill=True,
                guide=this,
                title="Variable cardinality chart",
                text="""
                Each horizontal bar shows the distinct observed values found for a variable. Colors identify type/cardinality combinations 
                that deserve review. Hover for the current role, data type, missingness, uniqueness, analysis basis, and explanation.
                """,
                position="left",
            ),
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.span(
                "Cardinality interpretation table",
                class_="text-primary text-center d-block",
            ),
            ui.output_ui(
                id="Profile",
                guide=this, title="Cardinality interpretation table", position="left",
                text="""
                The table combines the current role, cardinality, missingness, concentration, data type, and suggested considerations for 
                the subsequent Role Assignment card. Row colors match the chart findings. It reports roles but does not assign or validate them.
                """,
            ),
        )

    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(id="Busy"),
            ui.output_ui(
                id="Check",
                guide=this, title="Cardinality summary", position="top",
                text="Summarizes the variables needing review and states whether the analysis used all observations or a deterministic sample.",
            ),
            class_="text-center",
        )

    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_checkbox(
                id="ShowThresholds", label="Show cardinality thresholds", value=True,
                guide=this, title="Show thresholds", position="left",
                text=(
                    "Draws the low- and high-cardinality reference lines. These "
                    "are screening guides rather than universal statistical limits."
                ),
            ),
            ui.input_checkbox(
                id="Logarithmic", label="Logarithmic cardinality axis", value=True,
                guide=this, title="Logarithmic axis", position="left",
                text=(
                    "Uses a logarithmic horizontal axis so variables with a few "
                    "levels remain visible beside variables with thousands of values."
                ),
            ),
            ui.input_slider(
                id="Thresholds", label="Cardinality thresholds", min=2, max=100,
                value=(DEFAULT_LOW_CARDINALITY, DEFAULT_HIGH_CARDINALITY), step=1,
                guide=this, title="Cardinality thresholds", position="left",
                text=(
                    "The lower value matches Role Assignment's default maximum "
                    "cardinality for low-cardinality specialized roles. The upper "
                    "value identifies nominal variables whose number of levels may "
                    "require an identifier, text, or feature-engineering decision."
                ),
            ),
            ui.input_slider(
                id="MaxObs", label="Maximum observations to analyze", min=3, max=7, value=5, ticks=True, pre="10^",
                guide=this, title="Maximum observations", position="left",
                text=(
                    "Above 10^n rows the chart uses a deterministic random sample. "
                    "Low-cardinality conclusions use bounded full-data checks, and "
                    "complete-data uniqueness is verified before a variable is called unique."
                ),
            ),
            ui.input_select(
                id="Ordering", label="Variable ordering", selected="cardinality",
                choices={
                    "cardinality": "Cardinality",
                    "finding": "Finding",
                    "original": "Original column order",
                },
                guide=this, title="Variable ordering", position="left",
                text=(
                    "Orders bars by distinct values, by review priority, or by the "
                    "incoming data's column order. This changes presentation only."
                ),
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
            thresholds = tuple(input.Thresholds() or ())
            req(len(thresholds) == 2)
            low, high = sorted(int(value) for value in thresholds)
            return {
                "maximum_observations": 10 ** int(input.MaxObs()),
                "low_threshold": low,
                "high_threshold": high,
                "ordering": str(input.Ordering()),
                "logarithmic": bool(input.Logarithmic()),
                "show_thresholds": bool(input.ShowThresholds()),
            }

        @busy.track("Profiling variable cardinality…")
        @reactive.extended_task
        async def Calculate(data: proxy_data, options: dict[str, object]):
            analysis_options = {
                key: options[key]
                for key in (
                    "maximum_observations", "low_threshold", "high_threshold",
                )
            }
            return await asyncio.to_thread(
                _analyse_cardinality, data, **analysis_options,
            )

        @this.suspendable()
        def StartAnalysis():
            # proxy_data frames are read-only to card consumers. Avoid cloning a
            # potentially multi-million-row frame merely to profile it.
            Calculate.invoke(incomingproxy_data(), Options())

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
        def Chart():
            analysis = Analysis()
            options = Options()
            full_screen = bool(this.isFullScreen())
            figure = _cardinality_figure(
                analysis,
                ordering=options["ordering"],
                logarithmic=options["logarithmic"],
                show_thresholds=options["show_thresholds"],
                low_threshold=options["low_threshold"],
                high_threshold=options["high_threshold"],
                full_screen=full_screen,
            )
            widget = go.FigureWidget(figure)
            widget._config = getattr(widget, "_config", {}) | {
                "displayModeBar": full_screen,
                "displaylogo": False,
                "responsive": True,
            }
            return widget

        @output
        @render.ui
        def Profile():
            # Keep the data-frame output dynamic. Static data-frame bindings can
            # be initialized a second time when Sortable moves and rebinds a
            # card. Waiting for the analysis before inserting the inner output
            # follows the established workaround used by var_dissimilar.
            req(Analysis() is not None)
            return ui.output_data_frame(id="ProfileTable")

        @output
        @render.data_frame
        def ProfileTable():
            table = _ordered_profiles(Analysis().profiles, "finding")
            visible = table.drop(columns=["Distinct value"], errors="ignore")
            return render.DataTable(
                visible,
                width="100%",
                height="98%",
                styles=_profile_row_styles(visible),
            )

        @output
        @render.ui
        def Check():
            analysis = Analysis()
            table = analysis.profiles
            strong = int(table["Finding"].eq("Strong review").sum()) if not table.empty else 0
            review = int(table["Finding"].eq("Review").sum()) if not table.empty else 0
            expected = int(table["Finding"].eq("Expected").sum()) if not table.empty else 0
            unassessed = int(table["Finding"].eq("Not assessed").sum()) if not table.empty else 0
            basis = (
                f"a deterministic sample of {analysis.sampled_observations:,} from "
                f"{analysis.source_observations:,} observations"
                if analysis.sampled
                else f"all {analysis.source_observations:,} observations"
            )
            return ui.span(
                f"Strong review: {strong}; review: {review}; expected: {expected}; "
                f"not assessed: {unassessed}. Based on {basis}.",
                class_="text-warning" if strong or review else "text-success",
            )

        session.on_ended(Calculate.cancel)

        return incomingproxy_data

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
