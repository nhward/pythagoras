from __future__ import annotations

import json
import os
import sys
import textwrap
from collections.abc import Mapping
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
from shiny import render, req, ui
from shinywidgets import render_widget

JOURNEY_COLUMNS = [
    "Step",
    "Stage",
    "Attempted",
    "Card",
    "Operation",
    "Method",
    "Variables",
    "Parameters",
    "Input shape",
    "Output shape",
    "Row change",
    "Variable change",
]

STAGE_COLOURS = {
    "Source": "#6c757d",
    "Cleaning": "#2a9d8f",
    "Learning": "#e07a1f",
    "Preview": "#154c79",
}

STAGE_ROW_COLOURS = {
    "Source": "rgba(108, 117, 125, 0.12)",
    "Cleaning": "rgba(42, 157, 143, 0.14)",
    "Learning": "rgba(224, 122, 31, 0.14)",
    "Preview": "rgba(21, 76, 121, 0.12)",
}

HOVER_VALUE_MAX_LENGTH = 40
CARD_STEPS_PER_ROW = 6
FULL_SCREEN_STEPS_PER_ROW = 12
JOURNEY_ROW_HEIGHT = 175


def _shape_text(shape: tuple[int, int]) -> str:
    return f"{shape[0]:,} × {shape[1]:,}"


def _display_value(value: object) -> object:
    """Return a compact JSON-compatible representation for the table."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _display_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        shown = [_display_value(item) for item in values[:12]]
        if len(values) > 12:
            shown.append(f"… {len(values) - 12} more")
        return shown
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parameters_text(parameters: Mapping[str, object]) -> str:
    if not parameters:
        return ""
    changes = parameters.get("changes")
    if isinstance(changes, (list, tuple)):
        descriptions = []
        for change in changes:
            if not isinstance(change, Mapping):
                descriptions.append(str(change))
                continue
            variable = str(change.get("variable", "Variable"))
            details = []
            if change.get("new_name") is not None:
                details.append(f"rename to {change['new_name']}")
            if (
                change.get("original_type") is not None
                and change.get("new_type") is not None
            ):
                details.append(
                    f"type {change['original_type']} → {change['new_type']}"
                )
            if change.get("original_order") != change.get("new_order"):
                details.append(
                    f"order {change.get('original_order')} → {change.get('new_order')}"
                )
            descriptions.append(f"{variable}: {', '.join(details) or 'modified'}")
        return "\n".join(f"• {description}" for description in descriptions)
    return json.dumps(
        _display_value(parameters),
        ensure_ascii=False,
        sort_keys=True,
        separators=(", ", ": "),
    )


def _cleaning_variables(parameters: Mapping[str, object]) -> str:
    variables: list[str] = []
    direct = parameters.get("variable")
    if direct is not None:
        variables.append(str(direct))
    changes = parameters.get("changes")
    if isinstance(changes, (list, tuple)):
        for change in changes:
            if isinstance(change, Mapping) and change.get("variable") is not None:
                variables.append(str(change["variable"]))
    return ", ".join(dict.fromkeys(variables)) or "Dataset"


def _learning_variables(estimator: object) -> str:
    for attribute in ("columns", "eligible"):
        values = getattr(estimator, attribute, None)
        if values:
            return ", ".join(map(str, values))
    return "Dataset"


def _learning_parameters(estimator: object) -> dict[str, object]:
    if not hasattr(estimator, "get_params"):
        return {}
    omitted = {"columns", "eligible", "predictors"}
    return {
        name: value
        for name, value in estimator.get_params(deep=False).items()
        if name not in omitted
    }


def _journey_table(data: proxy_data) -> pd.DataFrame:
    """Describe the source, cleaning, learning, and current preview in order."""
    records = data.cleaning_records
    initial_shape = records[0].input_shape if records else data.clean_frame.shape
    rows: list[dict[str, object]] = [{
        "Step": 0,
        "Stage": "Source",
        "Attempted": "",
        "Card": "data_import",
        "Operation": "Imported source data",
        "Method": "",
        "Variables": f"{initial_shape[1]:,} variables",
        "Parameters": _parameters_text({"name": data.name}) if data.name else "",
        "Input shape": "",
        "Output shape": _shape_text(initial_shape),
        "Row change": 0,
        "Variable change": 0,
    }]

    step = 1
    if data.processing_records:
        for record in data.processing_records:
            variables = ", ".join(record.variables)
            if not variables and record.stage == "Cleaning":
                variables = _cleaning_variables(record.parameters)
            rows.append({
                "Step": step,
                "Stage": record.stage,
                "Attempted": "Yes" if record.attempted else "No",
                "Card": record.card,
                "Operation": record.operation,
                "Method": record.method,
                "Variables": variables or "Dataset",
                "Parameters": _parameters_text(record.parameters),
                "Input shape": _shape_text(record.input_shape),
                "Output shape": _shape_text(record.output_shape),
                "Row change": record.output_shape[0] - record.input_shape[0],
                "Variable change": record.output_shape[1] - record.input_shape[1],
            })
            step += 1
    else:
        # Compatibility for proxies created before processing records existed.
        shape = data.clean_frame.shape
        for record in records:
            rows.append({
                "Step": step,
                "Stage": "Cleaning",
                "Attempted": "Yes",
                "Card": record.card,
                "Operation": record.operation,
                "Method": "Materialised operation",
                "Variables": _cleaning_variables(record.parameters),
                "Parameters": _parameters_text(record.parameters),
                "Input shape": _shape_text(record.input_shape),
                "Output shape": _shape_text(record.output_shape),
                "Row change": record.output_shape[0] - record.input_shape[0],
                "Variable change": record.output_shape[1] - record.input_shape[1],
            })
            step += 1
        pipeline = data.pipeline
        if pipeline is not None:
            for name, estimator in pipeline.steps:
                rows.append({
                    "Step": step,
                    "Stage": "Learning",
                    "Attempted": "Yes",
                    "Card": name,
                    "Operation": name.replace("_", " ").title(),
                    "Method": type(estimator).__name__,
                    "Variables": _learning_variables(estimator),
                    "Parameters": _parameters_text(
                        _learning_parameters(estimator)
                    ),
                    "Input shape": _shape_text(shape),
                    "Output shape": _shape_text(shape),
                    "Row change": 0,
                    "Variable change": 0,
                })
                step += 1

    rows.append({
        "Step": step,
        "Stage": "Preview",
        "Attempted": "",
        "Card": "",
        "Operation": "Current materialised preview",
        "Method": "Full-data preview only",
        "Variables": f"{data.shape[1]:,} variables",
        "Parameters": "The stored learning pipeline remains unfitted",
        "Input shape": _shape_text(data.clean_frame.shape),
        "Output shape": _shape_text(data.shape),
        "Row change": data.shape[0] - data.clean_frame.shape[0],
        "Variable change": data.shape[1] - data.clean_frame.shape[1],
    })
    return pd.DataFrame(rows, columns=JOURNEY_COLUMNS)


def _visible_journey(table: pd.DataFrame, *, hide_inactive: bool) -> pd.DataFrame:
    """Optionally hide disabled steps without mistaking no-ops for inactivity."""
    if not hide_inactive or "Attempted" not in table.columns:
        return table.reset_index(drop=True)
    return table.loc[~table["Attempted"].eq("No")].reset_index(drop=True)


def _short_label(value: str, width: int = 18) -> str:
    lines = textwrap.wrap(str(value), width=width, max_lines=2, placeholder="…")
    return "<br>".join(lines)


def _hover_value(
    value: object,
    max_length: int = HOVER_VALUE_MAX_LENGTH,
) -> str:
    """Abbreviate a table value for a single readable hover-text line."""
    text = " ".join(str(value).split())
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 3].rstrip()}..."


def _journey_positions(
    step_count: int,
    *,
    steps_per_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Place sequential steps in alternating horizontal rows."""
    indices = np.arange(step_count, dtype=int)
    rows = indices // steps_per_row
    offsets = indices % steps_per_row
    x = np.where(rows % 2 == 0, offsets, steps_per_row - 1 - offsets)
    return x.astype(float), -rows.astype(float)


def _journey_figure(table: pd.DataFrame, *, full_screen: bool = False) -> go.Figure:
    """Draw a responsive processing flow that snakes across multiple rows."""
    if table.empty:
        return Card.empty_figure("No data journey is available")

    figure = go.Figure()
    steps_per_row = (
        FULL_SCREEN_STEPS_PER_ROW if full_screen else CARD_STEPS_PER_ROW
    )
    x, y = _journey_positions(
        len(table), steps_per_row=steps_per_row,
    )
    row_count = int(np.ceil(len(table) / steps_per_row))
    figure.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"color": "rgba(70,80,90,0.45)", "width": 4},
        hoverinfo="skip",
        showlegend=False,
    ))

    hover_columns = [column for column in JOURNEY_COLUMNS if column != "Step"]
    hover_template = "<br>".join(
        ["<b>Step %{text}</b>"]
        + [
            f"{column}: %{{customdata[{index}]}}"
            for index, column in enumerate(hover_columns)
        ]
        + ["<extra></extra>"]
    )
    marker_size = 64 if full_screen else 52
    for stage, colour in STAGE_COLOURS.items():
        selected = table["Stage"].eq(stage)
        if not selected.any():
            continue
        subset = table.loc[selected]
        selected_positions = np.flatnonzero(selected.to_numpy())
        figure.add_trace(go.Scatter(
            x=x[selected_positions],
            y=y[selected_positions],
            mode="markers+text",
            marker={
                "symbol": "square",
                "size": marker_size,
                "color": colour,
                "line": {"color": "white", "width": 2},
            },
            text=subset["Step"].astype(str),
            textfont={"color": "white", "size": 13},
            customdata=np.asarray([
                [_hover_value(value) for value in row]
                for row in subset[hover_columns]
                .fillna("")
                .itertuples(index=False, name=None)
            ], dtype=object),
            hovertemplate=hover_template,
            name=stage,
        ))

    for index, row in table.iterrows():
        position = table.index.get_loc(index)
        figure.add_annotation(
            x=x[position],
            y=y[position] - 0.27,
            text=_short_label(row["Operation"]),
            showarrow=False,
            align="center",
            font={"size": 11 if not full_screen else 13, "color": "#334155"},
        )
        if position:
            delta_x = x[position] - x[position - 1]
            delta_y = y[position] - y[position - 1]
            distance = float(np.hypot(delta_x, delta_y))
            unit_x = delta_x / distance
            unit_y = delta_y / distance
            figure.add_annotation(
                x=x[position] - 0.34 * unit_x,
                y=y[position] - 0.34 * unit_y,
                ax=x[position - 1] + 0.34 * unit_x,
                ay=y[position - 1] + 0.34 * unit_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.1,
                arrowwidth=2,
                arrowcolor="#59636e",
            )

    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(300, row_count * JOURNEY_ROW_HEIGHT + 105),
        margin={"l": 25, "r": 25, "t": 15, "b": 75},
        hovermode="closest",
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.32,
            "yanchor": "top",
        },
        xaxis={
            "range": [-0.6, max(float(x.max()), 0.0) + 0.6],
            "visible": False,
            "fixedrange": not full_screen,
        },
        yaxis={
            "range": [-row_count + 0.45, 0.45],
            "visible": False,
            "fixedrange": True,
        },
    )
    return figure


def _row_styles(table: pd.DataFrame) -> list[dict[str, object]]:
    styles = []
    for stage, colour in STAGE_ROW_COLOURS.items():
        rows = table.index[table["Stage"].eq(stage)].tolist()
        if rows:
            styles.append({
                "rows": rows,
                "style": {"background-color": colour},
            })
    return styles


def instance():
    """Create the immutable data-provenance card."""
    this = Card(file=__file__, mutable=False)
    this.long_name = "Data journey"
    this.description = "This card shows the order of cleaning operations, learned transformations and learned models."

    def front():
        return ui.TagList(
            ui.span(
                "Data cleaning and learning journey",
                class_="text-primary text-center d-block",
            ),
            ui.div(
                shinywidgets.output_widget(
                    id="JourneyChart", width="100%", height="auto", fill=False,
                    guide=this, title="Data journey", position="left",
                    text="Follow each row in alternating directions. Green boxes are materialised cleaning operations; orange boxes are learned sklearn steps."
                ),
                class_="journey-chart-scroll html-fill-item",
            ),
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.span(
                "Data journey details",
                class_="text-primary text-center d-block",
            ),
            ui.output_data_frame(
                id="JourneyTable",
                guide=this, title="Data journey table", position="left",
                text="Each row corresponds to a chart box and reports its card, method, affected variables, parameters, and shape change."
            ),
        )

    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(
                id="Status",
                guide=this, title="Summary", position="left",
                text="A summary of the data-journey steps"
            ),
            class_="html-fill-container html-fill-item text-center",
        )

    this.footer = footer

    def settings():
        return ui.input_checkbox(
            id="HideInactive", label="Hide steps that were not enabled", value=True,
            guide=this, title="Hide inactive steps", position="left",
            text="Hide operations whose data-changing control was not enabled. An enabled operation remains visible even when the data did not require any change."
        )

    this.settings = settings

    def server(input, output, session):
        @this.suspendable(calc=True)
        def incomingproxy_data():
            return this.input_data()

        @this.suspendable(calc=True)
        @this.record_code
        def JourneyData():
            return _journey_table(incomingproxy_data())

        @this.suspendable(calc=True)
        def VisibleJourney():
            return _visible_journey(
                JourneyData(), hide_inactive=bool(input.HideInactive()),
            )

        @output
        @render_widget
        def JourneyChart():
            full_screen = bool(this.isFullScreen())
            figure = _journey_figure(VisibleJourney(), full_screen=full_screen)
            widget = go.FigureWidget(figure)
            widget._config = getattr(widget, "_config", {}) | {
                "displayModeBar": full_screen,
                "displaylogo": False,
                "responsive": True,
            }
            return widget

        @output
        @render.data_frame
        def JourneyTable():
            table = VisibleJourney()
            return render.DataGrid(
                table,
                filters=True,
                summary=True,
                width="100%",
                height="98%",
                styles=_row_styles(table),
            )

        @output
        @render.ui
        def Status():
            table = VisibleJourney()
            cleaning = int(table["Stage"].eq("Cleaning").sum())
            learning = int(table["Stage"].eq("Learning").sum())
            cleaning_noun = "step" if cleaning == 1 else "steps"
            learning_noun = "step" if learning == 1 else "steps"
            return ui.span(
                f"{cleaning} cleaning {cleaning_noun}; "
                f"{learning} learning {learning_noun}.",
                class_="text-primary",
            )

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    frame = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=frame, _name="Ass2"))
    this.run()
