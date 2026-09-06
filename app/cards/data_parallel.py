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
from shiny import reactive, render, ui
from shinywidgets import render_widget

NO_COLOUR = "__none__"
MAX_CATEGORICAL_LEVELS = 15
DEFAULT_AXES = 8
SAMPLE_SEED = 1729


@dataclass(frozen=True)
class ParallelData:
    frame: pd.DataFrame
    dimensions: list[dict[str, object]]
    colour: dict[str, object] | None
    identities: list[str]
    complete_observations: int
    source_observations: int


def _column_kind(series: pd.Series) -> str:
    """Classify scalar columns according to their parallel-axis encoding."""
    dtype = series.dtype
    if is_geometry(dtype):
        return "geometry"
    if is_list(dtype):
        return "list"
    if is_cyclic(dtype):
        return "cyclic"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
        return "numeric"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "categorical"
    return "unsupported"


def _eligible_columns(
    frame: pd.DataFrame,
    *,
    maximum_levels: int = MAX_CATEGORICAL_LEVELS,
) -> tuple[list[str], dict[str, str]]:
    """Return columns Plotly can encode and reasons for exclusions."""
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    for column in frame.columns:
        name = str(column)
        kind = _column_kind(frame[column])
        if name.startswith(Card.SHADOW_PREFIX):
            excluded[name] = "Shadow variable"
        elif kind in {"geometry", "list", "cyclic", "unsupported"}:
            excluded[name] = f"{kind.title()} variables cannot be represented"
        elif frame[column].notna().sum() == 0:
            excluded[name] = "No observed values"
        elif kind == "categorical":
            try:
                levels = int(frame[column].nunique(dropna=True))
            except TypeError:
                excluded[name] = "Values are not scalar"
                continue
            if levels > maximum_levels:
                excluded[name] = f"More than {maximum_levels} observed levels"
            else:
                eligible.append(name)
        else:
            eligible.append(name)
    return eligible, excluded


def _finite_complete_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop missing values and non-finite numeric values from selected columns."""
    if frame.empty:
        return frame.copy()
    keep = frame.notna().all(axis=1)
    for column in frame.columns:
        series = frame[column]
        if _column_kind(series) == "numeric":
            numeric = pd.to_numeric(series, errors="coerce").to_numpy(
                dtype=float, na_value=np.nan,
            )
            keep &= np.isfinite(numeric)
    return frame.loc[keep].copy()


def _categorical_values(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed = set(series.dropna().tolist())
        levels = [value for value in series.cat.categories if value in observed]
    elif pd.api.types.is_bool_dtype(series.dtype):
        levels = [value for value in (False, True) if (series == value).any()]
    else:
        # String conversion makes mixed scalar object columns deterministic and
        # gives the chart readable tick labels.
        levels = list(dict.fromkeys(series.astype(str).tolist()))
        lookup = {value: index for index, value in enumerate(levels)}
        return series.astype(str).map(lookup).to_numpy(dtype=float), levels
    lookup = {value: index for index, value in enumerate(levels)}
    return series.map(lookup).to_numpy(dtype=float), [str(value) for value in levels]


def _datetime_values(series: pd.Series) -> tuple[np.ndarray, list[float], list[str]]:
    timestamps = pd.to_datetime(series)
    # Pandas may retain second, millisecond, microsecond, or nanosecond storage.
    # Normalising explicitly prevents the physical dtype unit from changing the
    # dates Plotly receives.
    values = (
        timestamps.to_numpy(dtype="datetime64[ns]")
        .astype("int64")
        .astype(float)
        / 1_000_000_000
    )
    low = float(values.min())
    high = float(values.max())
    tick_values = np.linspace(low, high, min(5, len(np.unique(values)))).tolist()
    tick_labels = [
        pd.to_datetime(value, unit="s").strftime("%Y-%m-%d\n%H:%M")
        for value in tick_values
    ]
    return values, tick_values, tick_labels


def _encode_series(series: pd.Series, label: str) -> tuple[dict[str, object], np.ndarray]:
    """Build one Plotly parcoords dimension and return its numeric values."""
    kind = _column_kind(series)
    dimension: dict[str, object] = {"label": label}
    if kind == "numeric":
        values = pd.to_numeric(series).to_numpy(dtype=float, na_value=np.nan)
    elif kind == "datetime":
        values, tick_values, tick_labels = _datetime_values(series)
        dimension.update(tickvals=tick_values, ticktext=tick_labels)
    else:
        values, levels = _categorical_values(series)
        ticks = list(range(len(levels)))
        dimension.update(tickvals=ticks, ticktext=levels)
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        padding = max(abs(low) * 0.05, 0.5)
        dimension["range"] = [low - padding, high + padding]
    else:
        dimension["range"] = [low, high]
    dimension["values"] = values
    return dimension, values


def _colour_line(
    series: pd.Series,
    label: str,
    *,
    show_scale: bool,
) -> dict[str, object]:
    kind = _column_kind(series)
    line: dict[str, object] = {
        "colorscale": "Turbo",
        "showscale": show_scale,
    }
    if kind == "numeric":
        values = pd.to_numeric(series).to_numpy(dtype=float, na_value=np.nan)
    elif kind == "datetime":
        values, tick_values, tick_labels = _datetime_values(series)
        line["colorbar"] = {
            "title": {"text": label},
            "tickvals": tick_values,
            "ticktext": tick_labels,
            "thickness": 12,
        }
    else:
        values, levels = _categorical_values(series)
        ticks = list(range(len(levels)))
        line["colorbar"] = {
            "title": {"text": label},
            "tickvals": ticks,
            "ticktext": levels,
            "thickness": 12,
        }
        if len(levels) > 1:
            colours = [
                colour
                for colour in (
                    "#154c79", "#d95f02", "#1b9e77", "#7570b3", "#e7298a",
                    "#66a61e", "#e6ab02", "#a6761d", "#1f78b4", "#b2df8a",
                    "#fb9a99", "#cab2d6", "#fdbf6f", "#6a3d9a", "#b15928",
                )[:len(levels)]
            ]
            last = len(colours) - 1
            line["colorscale"] = [
                point
                for index, colour in enumerate(colours)
                for point in (
                    [max(0.0, (index - 0.5) / last), colour],
                    [min(1.0, (index + 0.5) / last), colour],
                )
            ]
    line["color"] = values
    line["cmin"] = float(np.min(values))
    line["cmax"] = float(np.max(values))
    if line["cmin"] == line["cmax"]:
        line["cmax"] = line["cmin"] + 1.0
    line.setdefault("colorbar", {"title": {"text": label}, "thickness": 12})
    return line


def _observation_identities(
    data: proxy_data,
    positions: list[int] | np.ndarray | None = None,
) -> list[str]:
    """Identify source rows by assigned key columns, with row-number fallback."""
    source = data.frame
    identifier_role = data.role_map.columns_with_role(Role.IDENTIFIER)
    identifiers = [column for column in source.columns if column in identifier_role]
    if positions is None:
        positions = np.arange(len(source))
    labels: list[str] = []
    for position in positions:
        row = source.iloc[int(position)]
        parts: list[str] = []
        complete_key = bool(identifiers)
        for column in identifiers:
            value = row[column]
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
                complete_key = False
                break
            if not isinstance(missing, (bool, np.bool_)):
                complete_key = False
                break
            parts.append(f"{column} = {value}")
        labels.append(
            "; ".join(parts) if complete_key else f"Row {int(position) + 1}"
        )
    return labels


def _prepare_parallel_data(
    data: proxy_data,
    variables: list[str],
    colour_variable: str,
    maximum_observations: int,
    *,
    show_colour_scale: bool,
) -> ParallelData:
    source = data.frame
    required = list(dict.fromkeys(
        variables + ([colour_variable] if colour_variable != NO_COLOUR else [])
    ))
    available = [column for column in required if column in source.columns]
    # A fresh positional index keeps identities aligned even when the source
    # index contains duplicates.
    frame = pd.DataFrame(source.loc[:, available]).reset_index(drop=True)
    frame = _finite_complete_cases(frame)
    complete_observations = len(frame)
    if len(frame) > maximum_observations:
        positions = np.sort(
            np.random.default_rng(SAMPLE_SEED).choice(
                len(frame), size=maximum_observations, replace=False,
            )
        )
        frame = frame.iloc[positions].copy()
    identities = _observation_identities(
        data, positions=frame.index.to_numpy(dtype=int),
    )
    dimensions = [
        _encode_series(frame[column], str(column))[0]
        for column in variables
        if column in frame.columns and not frame.empty
    ]
    colour = None
    if colour_variable != NO_COLOUR and colour_variable in frame and not frame.empty:
        colour = _colour_line(
            frame[colour_variable], str(colour_variable),
            show_scale=show_colour_scale,
        )
    return ParallelData(
        frame=frame,
        dimensions=dimensions,
        colour=colour,
        identities=identities,
        complete_observations=complete_observations,
        source_observations=len(source),
    )


def _parallel_figure(
    data: ParallelData,
    *,
    full_screen: bool,
) -> go.Figure:
    if data.frame.empty:
        return Card.empty_figure("No observations are complete for the selected variables")
    if len(data.dimensions) < 2:
        return Card.empty_figure("Select at least two compatible variables")
    trace_options: dict[str, object] = {
        "dimensions": data.dimensions,
        # Plotly parcoords does not currently emit native per-line hover events.
        # The client-side hover layer reads these public per-datum labels.
        "customdata": data.identities,
    }
    if data.colour is not None:
        trace_options["line"] = data.colour
    figure = go.Figure(go.Parcoords(**trace_options))
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        # background not used: plot_bgcolor="#bbd6f8",
        margin={
            "l": 65 if full_screen else 45,
            "r": 95 if data.colour is not None and full_screen else 35,
            "t": 35,
            "b": 30,
        },
        font={"size": 13 if full_screen else 10},
    )
    return figure


def instance():
    """Create the immutable parallel-coordinates card."""
    this = Card(file=__file__, mutable=False)
    this.long_name = "Parallel coordinates"
    this.description = (
        "This card compares observations across several variables in a "
        "parallel-coordinates chart. Each line represents one complete observation."
    )

    def front():
        return ui.TagList(
            ui.span(
                "Parallel coordinates",
                class_="text-primary text-center d-block",
            ),
            ui.div(
                ui.div(
                    ui.tags.button(
                        "Clear comparison",
                        type="button",
                        class_="btn btn-sm btn-outline-secondary parallel-clear-comparison",
                        hidden=True,
                    ),
                    class_="parallel-chart-controls",
                ),
                shinywidgets.output_widget(
                    id="Chart",
                    fill=True,
                    guide=this,
                    title="Parallel-coordinates chart",
                    text=(
                        "Each line is an observation spanning the selected variable "
                        "axes. Hover near a line to identify it and click to pin it for "
                        "comparison. Drag vertically on an axis to filter lines, "
                        "double-click to clear a filter, and drag an axis horizontally "
                        "to reorder it. Only observations complete across the displayed "
                        "and color variables are included."
                    ),
                    position="left",
                ),
                ui.div(
                    hidden=True,
                    class_="parallel-hover-tooltip",
                    role="status",
                ),
                ui.div(
                    hidden=True,
                    class_="parallel-comparison-label",
                    role="status",
                ),
                class_="parallel-hover-host html-fill-container html-fill-item",
                data_parallel_hover="true",
            ),
        )

    this.front = front

    def footer():
        return ui.div(
            ui.output_ui(id="Check"),
            class_="text-center",
        )

    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_selectize(
                id="Variables", label="Variables to display", choices=[], selected=[], multiple=True, options={"plugins": ["remove_button"]},
                guide=this, title="Variables", position="left",
                text= f"Select the vertical axes. Numeric, Boolean, date/time, and categorical variables with no more than {MAX_CATEGORICAL_LEVELS} observed levels can be displayed."
            ),
            ui.input_select(
                id="Colour", label="Color lines by", choices={NO_COLOUR: "No color variable"}, selected=NO_COLOUR,
                guide=this, title="Color variable", position="left",
                text="Optionally colors every observation line by another compatible variable. The scale is shown in full-screen mode; this changes only the chart, not the exported data."
            ),
            ui.input_slider(
                id="MaxObs", label="Maximum observations to chart", min=3, max=7, value=4, ticks=True, pre="10^",
                guide=this, text="Sets a logarithmic cap of 10^n complete observations drawn by deterministic random sampling. Raising it improves coverage but increases chart density and browser cost; no rows are removed downstream.", position="left",
            ),
        )

    this.settings = settings

    def server(input, output, session):

        @this.suspendable(calc=True)
        def incomingproxy_data():
            return this.input_data()

        @this.suspendable()
        def UpdateChoices():
            eligible, _ = _eligible_columns(incomingproxy_data().frame)
            with reactive.isolate():
                previous_variables = list(input.Variables() or [])
                previous_colour = input.Colour()
            selected = [name for name in previous_variables if name in eligible]
            if not selected:
                selected = eligible[:DEFAULT_AXES]
            colour = previous_colour if previous_colour in eligible else NO_COLOUR
            ui.update_selectize("Variables", choices=eligible, selected=selected)
            ui.update_select(
                "Colour",
                choices={NO_COLOUR: "No color variable"} | {
                    name: name for name in eligible
                },
                selected=colour,
            )

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def Options():
            return {
                "variables": list(input.Variables() or []),
                "colour": str(input.Colour() or NO_COLOUR),
                "maximum_observations": int(10**input.MaxObs())
            }

        @this.suspendable(calc=True)
        def PreparedData():
            options = Options()
            return _prepare_parallel_data(
                incomingproxy_data(),
                options["variables"],
                options["colour"],
                options["maximum_observations"],
                show_colour_scale=bool(this.isFullScreen()),
            )

        @output
        @render_widget
        def Chart():
            full_screen = bool(this.isFullScreen())
            figure = _parallel_figure(
                PreparedData(),
                full_screen=full_screen,
            )
            figure.update_layout(modebar={"orientation": "v"})
            widget = go.FigureWidget(figure)
            widget._config = getattr(widget, "_config", {}) | {
                "displayModeBar": full_screen,
                "displaylogo": False,
                "responsive": True,
            }
            return widget

        @output
        @render.ui
        def Check():
            prepared = PreparedData()
            variables = Options()["variables"]
            if len(variables) < 2:
                return ui.span(
                    "Select at least two compatible variables.",
                    class_="text-info",
                )
            omitted = prepared.source_observations - prepared.complete_observations
            sampled = prepared.complete_observations - len(prepared.frame)
            messages = [f"Showing {len(prepared.frame):,} observations"]
            if omitted:
                noun = "observation" if omitted == 1 else "observations"
                messages.append(f"{omitted:,} incomplete {noun} omitted")
            if sampled:
                noun = "observation" if sampled == 1 else "observations"
                messages.append(f"{sampled:,} complete {noun} sampled out")
            return ui.span(
                "; ".join(messages) + ".",
                class_="text-warning" if omitted or sampled else "text-success",
            )

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
