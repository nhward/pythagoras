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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from module import Module
from plotly.subplots import make_subplots
from proxy_data import proxy_data
from shiny import render, req, ui
from shinywidgets import render_widget

INTERSECTION_COLUMNS = [
    "Missing variables",
    "Degree",
    "Observations",
    "Proportion",
]


def _missing_variables(frame: pd.DataFrame) -> list[str]:
    """Return incomplete columns, ordered by missing count then name."""
    counts = frame.isna().sum()
    columns = [column for column in frame.columns if counts[column] > 0]
    return sorted(columns, key=lambda column: (-int(counts[column]), str(column).casefold()))


def _intersection_counts(
    frame: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Count exact missingness combinations, excluding complete observations."""
    if not variables:
        return pd.DataFrame(columns=INTERSECTION_COLUMNS)
    indicators = frame.loc[:, variables].isna()
    indicators = indicators.loc[indicators.any(axis=1)]
    if indicators.empty:
        return pd.DataFrame(columns=INTERSECTION_COLUMNS)
    grouped = indicators.groupby(variables, observed=True, sort=False).size()
    rows = []
    for membership, count in grouped.items():
        membership = membership if isinstance(membership, tuple) else (membership,)
        selected = tuple(
            variable
            for variable, present in zip(variables, membership)
            if bool(present)
        )
        rows.append({
            "Missing variables": ", ".join(map(str, selected)),
            "Degree": len(selected),
            "Observations": int(count),
            "Proportion": float(count) / len(frame) if len(frame) else 0.0,
            "_membership": selected,
        })
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["Observations", "Degree", "Missing variables"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def _select_intersections(
    intersections: pd.DataFrame,
    *,
    maximum: int,
    minimum_count: int,
) -> pd.DataFrame:
    if intersections.empty:
        return intersections.copy()
    return intersections.loc[
        intersections["Observations"] >= max(1, int(minimum_count))
    ].head(max(1, int(maximum))).reset_index(drop=True)


def _upset_figure(
    frame: pd.DataFrame,
    variables: list[str],
    intersections: pd.DataFrame,
    *,
    full_screen: bool = False,
) -> go.Figure:
    """Construct an interactive UpSet chart using native Plotly traces."""
    if len(variables) < 2:
        return Card.empty_figure("At least two variables with missing values are required")
    if intersections.empty:
        return Card.empty_figure("No missingness intersections meet the settings")
    x = list(range(len(intersections)))
    y = list(range(len(variables)))
    variable_position = {variable: position for position, variable in enumerate(variables)}
    set_sizes = frame.loc[:, variables].isna().sum().astype(int)
    inactive_marker_size = 13 if full_screen else 8
    active_marker_size = 18 if full_screen else 11
    connector_width = 5 if full_screen else 3
    figure = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.56, 0.44],
        column_widths=[0.32, 0.68],
        horizontal_spacing=0.025,
        vertical_spacing=0.055,
        specs=[[None, {}], [{}, {}]],
    )
    figure.add_trace(
        go.Bar(
            x=x,
            y=intersections["Observations"],
            marker_color="#154c79",
            customdata=np.column_stack([
                intersections["Missing variables"],
                intersections["Proportion"],
            ]),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Observations: %{y:,}"
                "<br>Proportion: %{customdata[1]:.1%}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=set_sizes.to_numpy(),
            y=y,
            orientation="h",
            marker_color="#154c79",
            text=set_sizes.to_numpy(),
            textposition="outside",
            customdata=np.asarray(variables, dtype=object),
            hovertemplate="%{customdata}: %{x:,} missing<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    inactive_x = np.repeat(x, len(variables))
    inactive_y = np.tile(y, len(intersections))
    figure.add_trace(
        go.Scatter(
            x=inactive_x,
            y=inactive_y,
            mode="markers",
            marker={
                "size": inactive_marker_size,
                "color": "rgba(80,80,80,0.18)",
            },
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    for index, row in intersections.iterrows():
        positions = sorted(variable_position[value] for value in row["_membership"])
        if len(positions) > 1:
            figure.add_trace(
                go.Scatter(
                    x=[index, index],
                    y=[positions[0], positions[-1]],
                    mode="lines",
                    line={"color": "#7b3f00", "width": connector_width},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )
        figure.add_trace(
            go.Scatter(
                x=[index] * len(positions),
                y=positions,
                mode="markers",
                marker={"size": active_marker_size, "color": "#7b3f00"},
                text=[row["Missing variables"]] * len(positions),
                customdata=[row["Observations"]] * len(positions),
                hovertemplate=(
                    "<b>%{text}</b><br>Observations: %{customdata:,}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    # The bar columns align directly with the membership matrix below, so an
    # x-axis title adds no information and collides with the matrix in a card.
    figure.update_xaxes(showticklabels=False, ticks="", row=1, col=2)
    figure.update_yaxes(title_text="Intersection size", rangemode="tozero", row=1, col=2)
    figure.update_xaxes(
        title_text="Missing observations",
        autorange="reversed",
        rangemode="tozero",
        row=2,
        col=1,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=y,
        ticktext=[str(variable) for variable in variables],
        range=[len(variables) - 0.5, -0.5],
        row=2,
        col=1,
    )
    figure.update_xaxes(showticklabels=False, range=[-0.5, len(x) - 0.5], row=2, col=2)
    figure.update_yaxes(
        # Variable names are already supplied by the adjacent set-size plot.
        # Repeating them here makes the two lower subplots overlap.
        showticklabels=False,
        ticks="",
        range=[len(variables) - 0.5, -0.5],
        row=2,
        col=2,
    )
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#e5ecf6",
        margin={"l": 15, "r": 20, "t": 10, "b": 15},
        bargap=0.22,
        hovermode="closest",
    )
    return figure


def instance():
    """Create the immutable missingness-set card."""
    this = Card(file=__file__, mutable=False)
    this.long_name = "Missingness sets"
    this.description = "This card visualises which variables are missing together, both as individual sets and as exact intersections."

    def front():
        return ui.TagList(
            ui.span("Missingness sets", class_="text-primary text-center d-block"),
            shinywidgets.output_widget(
                id="Upset",
                fill=True,
                guide=this,
                title="Missingness UpSet chart",
                text=(
                    "The left bars count missing values in each variable. The top "
                    "bars count exact combinations, identified by the connected "
                    "dots beneath them."
                ),
                position="left",
            ),
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.span(
                "Missingness intersections",
                class_="text-primary text-center d-block",
            ),
            ui.output_ui(
                id="Table",
                guide=this,
                title="Missingness intersection table",
                text="Each row is one exact combination of variables missing in the same observations.",
                position="left",
            ),
        )

    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(id="Check"),
            id="X-Check",
            class_="html-fill-container html-fill-item text-center",
        )

    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_slider(
                id="MaxIntersections", label="Maximum intersections to chart", min=5, max=100, value=40, step=1,
                guide=this, text="Limits the top missingness combinations shown in the chart and table.", position="left",
            ),
            ui.input_slider(
                id="MaxVariables", label="Maximum variables to chart", min=2, max=30, value=15, step=1,
                guide=this, text="Keeps the most frequently missing variables when the data has many incomplete columns.", position="left",
            ),
            ui.input_numeric(
                id="MinCount", label="Minimum observations per intersection", value=1, min=1, step=1,
                guide=this, text="Hides exact missingness combinations occurring fewer times than this threshold.", position="left",
            ),
            ui.input_slider(
                id="MaxObs", label="Maximum observations to analyse", min=3, max=7, value=4, ticks=True, pre="10^",
                guide=this, text="Limits observations by random sampling to ensure responsiveness.", position="left",
            ),
        )

    this.settings = settings

    def server(input, output, session):
        
        @this.suspendable(calc=True)
        def incomingproxy_data():
            req(this._imports.is_set())
            return this._imports.get()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MaxObs():
            return 10**input.MaxObs()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MaxIntersections():
            return max(1, int(input.MaxIntersections()))

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MaxVariables():
            return max(2, int(input.MaxVariables()))

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinCount():
            return max(1, int(input.MinCount()))

        @this.suspendable(calc=True)
        @this.record_code
        def PreparedData():
            return incomingproxy_data().sample(
                n=MaxObs(), mode="random", keep_geometry=True
            )

        @this.suspendable(calc=True)
        @this.record_code
        def MissingVariables():
            frame = PreparedData().to_native()
            return _missing_variables(frame)[:MaxVariables()]

        @this.suspendable(calc=True)
        @this.record_code
        def Intersections():
            frame = PreparedData().to_native()
            counts = _intersection_counts(frame, MissingVariables())
            return _select_intersections(
                counts,
                maximum=MaxIntersections(),
                minimum_count=MinCount(),
            )

        @this.suspendable(calc=True)
        @this.record_code
        def IntersectionTable():
            table = Intersections().drop(columns="_membership", errors="ignore").copy()
            if not table.empty:
                table["Proportion"] = table["Proportion"].round(3)
            return table

        @output
        @render_widget
        def Upset():
            frame = PreparedData().to_native()
            full_screen = bool(this.isFullScreen())
            figure = _upset_figure(
                frame,
                MissingVariables(),
                Intersections(),
                full_screen=full_screen,
            )
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
            return render.DataTable(
                IntersectionTable(), width="100%", height="98%"
            )

        @output
        @render.ui
        def Check():
            all_missing = _missing_variables(PreparedData().to_native())
            shown = MissingVariables()
            if not all_missing:
                return ui.span(
                    "The data does not contain missing values.",
                    class_="text-success",
                )
            if len(all_missing) == 1:
                return ui.span(
                    "Only one variable has missing values; at least two are required for an UpSet chart.",
                    class_="text-info",
                )
            hidden = len(all_missing) - len(shown)
            suffix = f"; {hidden} less-frequently missing variables are hidden" if hidden else ""
            return ui.span(
                f"Showing {len(Intersections())} intersections across {len(shown)} missing variables{suffix}.",
                class_="text-primary",
            )

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
