from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

import asyncio
import math
from itertools import combinations

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

RESULT_COLUMNS = [
    "Differences tolerated",
    "Redundant row numbers",
    "Count",
]
MAX_COMBINATIONS = 50_000
EXCLUDED_ROLES = {Role.NONE, Role.GEOMETRY, Role.WEIGHTING, Role.IDENTIFIER}


def _eligible_columns(proxy: proxy_data) -> list[str]:
    """Select comparison columns, respecting roles used by the application."""
    return [
        column
        for column in proxy.columns
        if (
            not str(column).startswith(Card.SHADOW_PREFIX)
            and not (proxy.role_map.roles_for(column) & EXCLUDED_ROLES)
        )
    ]


def _round_significant(frame: pd.DataFrame, figures: int) -> pd.DataFrame:
    """Round floating-point columns to significant figures; preserve integers."""
    result = frame.copy()
    figures = max(1, int(figures))
    if figures >= 16:
        return result
    for column in result.columns:
        series = result[column]
        if not pd.api.types.is_float_dtype(series.dtype):
            continue
        values = series.to_numpy(dtype=float, copy=True)
        mask = np.isfinite(values) & (values != 0)
        if mask.any():
            magnitude = np.floor(np.log10(np.abs(values[mask])))
            scale = np.power(10.0, figures - 1 - magnitude)
            values[mask] = np.round(values[mask] * scale) / scale
        result[column] = values
    return result


def _combination_count(column_count: int, maximum_differences: int) -> int:
    maximum = min(max(0, int(maximum_differences)), max(0, column_count - 1))
    return sum(math.comb(column_count, difference) for difference in range(maximum + 1))


def _freeze(value):
    """Make common container values hashable without changing equality meaning."""
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _comparison_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column].dtype):
            result[column] = result[column].map(_freeze)
    return result


def _exact_duplicate_mask(proxy: proxy_data, significant_figures: int) -> np.ndarray:
    """Identify later exact duplicates using the card's comparison policy."""
    columns = _eligible_columns(proxy)
    if not columns:
        return np.zeros(len(proxy.to_native()), dtype=bool)
    comparison = _round_significant(
        proxy.to_native().loc[:, columns], significant_figures
    )
    comparison = _comparison_frame(comparison)
    return comparison.duplicated(keep="first").to_numpy()


def _deduplicate_proxy(
    proxy: proxy_data,
    significant_figures: int,
) -> proxy_data:
    """Return a cloned proxy with later exact duplicates removed."""
    duplicate = _exact_duplicate_mask(proxy, significant_figures)
    result = proxy.clone()
    result.data = result.to_native().iloc[~duplicate].copy()
    return result


def _duplicate_results(
    frame: pd.DataFrame,
    *,
    maximum_differences: int,
    maximum_combinations: int = MAX_COMBINATIONS,
) -> pd.DataFrame:
    """Classify redundant rows by their minimum number of column mismatches."""
    column_count = frame.shape[1]
    if column_count == 0 or len(frame) == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    maximum = min(max(0, int(maximum_differences)), column_count - 1)
    count = _combination_count(column_count, maximum)
    if count > maximum_combinations:
        raise ValueError(
            f"This setting requires {count:,} column combinations; reduce the "
            f"maximum differences below {maximum}."
        )

    comparable = _comparison_frame(frame).reset_index(drop=True)
    already_classified = np.zeros(len(comparable), dtype=bool)
    rows = []
    for difference in range(maximum + 1):
        duplicate = np.zeros(len(comparable), dtype=bool)
        width = column_count - difference
        for selected in combinations(comparable.columns, width):
            duplicate |= comparable.duplicated(
                subset=list(selected), keep="first"
            ).to_numpy()
        newly_classified = duplicate & ~already_classified
        positions = (np.flatnonzero(newly_classified) + 1).tolist()
        rows.append({
            "Differences tolerated": difference,
            "Redundant row numbers": ", ".join(map(str, positions)),
            "Count": len(positions),
        })
        already_classified |= newly_classified
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _duplicates_figure(results: pd.DataFrame, *, full_screen: bool = False) -> go.Figure:
    if results.empty:
        return Card.empty_figure("No variables are available for duplicate comparison")
    figure = go.Figure(go.Bar(
        x=results["Differences tolerated"].astype(str),
        y=results["Count"],
        marker_color="#154c79",
        customdata=results["Differences tolerated"],
        hovertemplate=(
            "%{y:,} redundant observation(s) first detected with "
            "%{customdata} difference(s) tolerated<extra></extra>"
        ),
    ))
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#bbd6f8",
        margin={"l": 45, "r": 30 if full_screen else 10, "t": 10, "b": 55},
        xaxis_title="Number of differences tolerated",
        yaxis_title="Redundant observations",
        yaxis={"rangemode": "tozero", "fixedrange": not full_screen},
        xaxis={"type": "category", "fixedrange": not full_screen},
        bargap=0.25,
        showlegend=False,
    )
    return figure


def instance():
    """Create the mutable observation-duplicates card."""
    this = Card(file=__file__, mutable=True)
    this.long_name = "Observation duplicates"
    this.description = "This card explores the diversity of observations of a dataset with respect to duplicates and near duplicates."
    
    def front():
        return ui.TagList(
            ui.output_ui(id="FrontTitle"),
            shinywidgets.output_widget(
                id="BarChart",
                fill=True,
                guide=this,
                title="Near-duplicate observations chart",
                text=(
                    "The zero bar contains exact duplicate rows. Later bars contain "
                    "new redundant rows first detected when that many variable "
                    "differences are tolerated."
                ),
                position="left",
            ),
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.output_ui(id="BackTitle"),
            ui.output_ui(
                id="Table",
                guide=this,
                title="Near-duplicate observation table",
                text=(
                    "Lists the one-based row numbers classified at each minimum "
                    "number of tolerated differences."
                ),
                position="left",
            ),
        )

    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(id="Busy"),
            ui.output_ui(id="Check"),
            ui.input_checkbox_group(
                id="RemoveExact", 
                label = "Remove",
                choices=["Exact duplicates"],
                inline=True,
                guide=this, title="Remove exact duplicates", position="top",
                text="Remove later observations that exactly duplicate an earlier observation under the current significant-figures setting.",
            ),
            class_ = "vertically-scrollable-footer", # class_="html-fill-container html-fill-item text-center",
        )

    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_slider(
                id="SignificantFigures", label="Significant figures used to compare numeric values", min=1, max=16, value=16, step=1,
                guide=this, text="Reducing this value rounds floating-point data and can make nearby numeric values compare as equal. Integers are unchanged.", position="left",
            ),
            ui.input_slider(
                id="MaxDifferences", label="Maximum number of differences tolerated", min=0, max=10, value=2, step=1,
                guide=this, text="Increasing this searches more column subsets and can become computationally expensive for wide data.", position="left",
            ),
        )

    this.settings = settings

    def server(input, output, session):
        busy = this.busy()

        @this.suspendable(calc=True)
        def incomingproxy_data():
            req(this._imports.is_set())
            return this._imports.get()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def SignificantFigures():
            return max(1, int(input.SignificantFigures()))

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MaxDifferences():
            return max(0, int(input.MaxDifferences()))

        @this.suspendable(calc=True)
        @this.record_code
        def TransformedData():
            proxy = incomingproxy_data()
            if not "Exact duplicates" in (input.RemoveExact() or []) :
                return proxy
            return _deduplicate_proxy(proxy, SignificantFigures())

        @this.suspendable(calc=True)
        @this.record_code
        def PreparedData():
            proxy = TransformedData()
            columns = _eligible_columns(proxy)
            frame = proxy.to_native().loc[:, columns]
            return _round_significant(frame, SignificantFigures())

        @this.suspendable(triggers=[TransformedData])
        def export():
            this._exports.set(TransformedData())

        @busy.track("Searching for duplicate and near-duplicate observations…")
        @reactive.extended_task
        async def CalculateDuplicates(
            frame: pd.DataFrame,
            maximum_differences: int,
        ) -> pd.DataFrame:
            return await asyncio.to_thread(
                _duplicate_results,
                frame,
                maximum_differences=maximum_differences,
            )

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @this.suspendable()
        def StartAnalysis():
            frame = PreparedData().copy()
            maximum = min(
                MaxDifferences(),
                max(0, frame.shape[1] - 1),
            )
            CalculateDuplicates.invoke(frame, maximum)

        @this.suspendable(calc=True)
        @this.record_code
        def Results():
            return CalculateDuplicates.result()

        @reactive.effect
        def LimitDifferences():
            column_count = len(_eligible_columns(incomingproxy_data()))
            maximum = min(10, max(0, column_count - 1))
            with reactive.isolate():
                selected = min(int(input.MaxDifferences()), maximum)
            ui.update_slider("MaxDifferences", max=maximum, value=selected)

        def _title(side: str):
            figures = SignificantFigures()
            suffix = "" if figures == 16 else f" at {figures} significant figures"
            return ui.span(
                f"Near-duplicate observations {side}{suffix}",
                class_="text-primary text-center d-block",
            )

        @output
        @render.ui
        def FrontTitle():
            return _title("chart")

        @output
        @render.ui
        def BackTitle():
            return _title("table")

        @output
        @render_widget
        def BarChart():
            full_screen = bool(this.isFullScreen())
            figure = _duplicates_figure(Results(), full_screen=full_screen)
            figure.update_layout(
                modebar={"orientation": "v"},
                modebar_remove=[
                    "select2d", "lasso2d", "toggleHover", "toggleSpikelines",
                    "hoverClosestCartesian", "hoverCompareCartesian",
                ],
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
        def Table():
            return ui.output_data_frame(id="Table2")

        @output
        @render.data_frame
        def Table2():
            return render.DataTable(Results(), width="100%", height="98%")


        @output
        @render.ui
        def Check():
            results = Results()
            source = incomingproxy_data()
            original_observations = len(source.to_native())
            observations = len(TransformedData().to_native())
            if results.empty:
                return ui.span(
                    "No variables are available for duplicate comparison.",
                    class_="text-info",
                )
            removed = original_observations - observations
            if "Exact duplicates" in (input.RemoveExact() or []) and removed:
                noun = "row" if removed == 1 else "rows"
                return ui.span(
                    f"Removed {removed} redundant exact-duplicate {noun}; "
                    f"{observations} observations remain.",
                    class_="text-warning",
                )
            exact = int(results.loc[
                results["Differences tolerated"] == 0, "Count"
            ].sum())
            if exact == 0:
                return ui.span(
                    f"There are no exact duplicate rows among {observations} observations.",
                    class_="text-success",
                )
            noun = "row" if exact == 1 else "rows"
            return ui.span(
                f"There are {exact} redundant exact-duplicate {noun} among "
                f"{observations} observations.",
                class_="text-warning",
            )

        session.on_ended(CalculateDuplicates.cancel)

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
