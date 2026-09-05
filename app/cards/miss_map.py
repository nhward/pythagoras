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
from roles import Role, RoleMap
from shiny import render, ui
from shinywidgets import render_widget

SPECIAL_ROLES = {
    Role.TARGET,
    Role.IDENTIFIER,
    Role.WEIGHTING,
    Role.STRATIFIER,
    Role.TREATMENT,
    Role.GEOMETRY,
    Role.SEQUENCE,
    Role.SENSITIVE,
    Role.PARTITION,
}


def _variable_summary(
    frame: pd.DataFrame,
    role_map: RoleMap,
    threshold: float,
) -> pd.DataFrame:
    """Summarise missingness and flag variables requiring attention."""
    rows = []
    denominator = len(frame)
    for column in frame.columns:
        count = int(frame[column].isna().sum())
        proportion = count / denominator if denominator else 0.0
        roles = role_map.roles_for(column)
        role_text = ", ".join(sorted(role.value.title() for role in roles))
        if count and roles & SPECIAL_ROLES:
            issue = "Missing values in a special-role variable"
        elif proportion > threshold:
            issue = "Excessive variable missingness"
        elif count:
            issue = "Missing values"
        else:
            issue = "Complete"
        rows.append({
            "Variable": str(column),
            "Role": role_text,
            "Missing": count,
            "Missing proportion": proportion,
            "Issue": issue,
        })
    return pd.DataFrame(rows)


def _excessive_observation_mask(
    frame: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    """Return rows whose proportion of missing fields is above the threshold."""
    if frame.shape[1] == 0:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame.isna().mean(axis=1) > threshold


def _remove_excessive_observations(
    data: proxy_data,
    threshold: float,
) -> proxy_data:
    frame = data.frame
    return data.with_cleaned_data(
        frame.loc[~_excessive_observation_mask(frame, threshold)].copy(),
        card="miss_map",
        operation="Remove excessively incomplete observations",
        parameters={"threshold": float(threshold)},
    )


def _remove_excessive_variables(
    data: proxy_data,
    threshold: float,
) -> proxy_data:
    """Remove columns whose missing proportion is above the threshold."""
    frame = data.frame
    keep = list(frame.columns[frame.isna().mean(axis=0) <= threshold])
    return data.with_cleaned_data(
        frame.loc[:, keep].copy(),
        card="miss_map",
        operation="Remove excessively incomplete variables",
        parameters={"threshold": float(threshold)},
    )


def _transform_data(
    data: proxy_data,
    *,
    remove_variables: bool,
    remove_observations: bool,
    variable_threshold: float,
    observation_threshold: float,
) -> proxy_data:
    """Apply selected removals and record the card as one cleaning step.

    Variable removal still precedes observation removal, but the two related
    controls describe one card operation in the data journey.
    """
    parameters = {
        "remove_variables": bool(remove_variables),
        "variable_threshold": float(variable_threshold),
        "remove_observations": bool(remove_observations),
        "observation_threshold": float(observation_threshold),
        "execution_order": "variables before observations",
    }
    operation = "Remove excessive missingness"
    if not remove_variables and not remove_observations:
        return data.with_inactive_step(
            stage="Cleaning",
            card="miss_map",
            operation=operation,
            parameters=parameters,
        )

    frame = data.frame.copy()
    if remove_variables:
        keep = list(
            frame.columns[
                frame.isna().mean(axis=0) <= float(variable_threshold)
            ]
        )
        frame = frame.loc[:, keep].copy()
    if remove_observations:
        excessive = _excessive_observation_mask(
            frame,
            float(observation_threshold),
        )
        frame = frame.loc[~excessive].copy()

    return data.with_cleaned_data(
        frame,
        card="miss_map",
        operation=operation,
        parameters=parameters,
    )


def _issues_table(
    frame: pd.DataFrame,
    role_map: RoleMap,
    variable_threshold: float,
    observation_threshold: float,
) -> pd.DataFrame:
    variables = _variable_summary(frame, role_map, variable_threshold)
    variable_issues = variables.loc[
        variables["Issue"].isin([
            "Missing values in a special-role variable",
            "Excessive variable missingness",
        ])
    ].copy()
    variable_issues.insert(0, "Scope", "Variable")
    variable_issues["Item"] = variable_issues.pop("Variable")
    variable_issues["Missing proportion"] = variable_issues[
        "Missing proportion"
    ].round(3)

    row_proportions = (
        frame.isna().mean(axis=1)
        if frame.shape[1]
        else pd.Series(0.0, index=frame.index)
    )
    excessive = row_proportions > observation_threshold
    observation_issues = pd.DataFrame({
        "Scope": "Observation",
        "Role": "",
        "Missing": frame.isna().sum(axis=1).loc[excessive].astype(int),
        "Missing proportion": row_proportions.loc[excessive].round(3),
        "Issue": "Excessive observation missingness",
    })
    observation_issues["Item"] = [
        str(position) for position in np.flatnonzero(excessive.to_numpy()) + 1
    ]
    columns = ["Scope", "Item", "Role", "Missing", "Missing proportion", "Issue"]
    return pd.concat(
        [variable_issues.reindex(columns=columns), observation_issues.reindex(columns=columns)],
        ignore_index=True,
    )


def _display_frame(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    # A fixed seed keeps the picture stable while controls unrelated to the data change.
    return frame.sample(n=maximum, random_state=1729).sort_index()


def _missingness_figure(
    frame: pd.DataFrame,
    *,
    hide_complete: bool,
    sort_variables: bool,
    variable_threshold: float,
    observation_threshold: float,
    show_thresholds: bool,
    full_screen: bool = False,
) -> go.Figure:
    """Draw a compact heatmap, with aligned marginal bars in full screen."""
    if frame.shape[0] == 0:
        return Card.empty_figure("No observations remain after applying the removal setting")
    if frame.shape[1] == 0:
        return Card.empty_figure("No variables are available")

    missing = frame.isna()
    counts = missing.sum(axis=0)
    columns = list(frame.columns)
    if hide_complete:
        columns = [column for column in columns if counts[column] > 0]
    if sort_variables:
        columns = sorted(columns, key=lambda column: (-int(counts[column]), str(column).casefold()))
    if not columns:
        return Card.empty_figure("The displayed data contains no missing values")

    matrix = missing.loc[:, columns].T.astype(int)
    x = list(range(len(frame)))
    y = list(range(len(columns)))
    observation_labels = [str(value) for value in frame.index]
    heatmap = go.Heatmap(
        z=matrix.to_numpy(),
        x=x,
        y=y,
        customdata=np.broadcast_to(np.asarray(observation_labels), matrix.shape),
        colorscale=[[0, "#f4e6b1"], [0.499, "#f4e6b1"], [0.5, "#7b3f00"], [1, "#7b3f00"]],
        showscale=False,
        hovertemplate="Variable: %{text}<br>Observation: %{customdata}<br>%{z}<extra></extra>",
        text=np.broadcast_to(np.asarray([str(column) for column in columns])[:, None], matrix.shape),
    )

    if full_screen:
        figure = make_subplots(
            rows=2,
            cols=2,
            row_heights=[0.78, 0.22],
            column_widths=[0.82, 0.18],
            horizontal_spacing=0.035,
            vertical_spacing=0.045,
            specs=[[{}, {}], [{}, None]],
        )
        figure.add_trace(heatmap, row=1, col=1)
        variable_proportions = matrix.mean(axis=1)
        figure.add_trace(go.Bar(
            x=variable_proportions,
            y=y,
            orientation="h",
            marker_color="#154c79",
            hovertemplate="%{x:.1%} missing<extra></extra>",
            showlegend=False,
        ), row=1, col=2)
        # Hiding complete variables is a display choice only. Observation
        # missingness (and therefore its threshold) must retain every variable
        # in the transformed data, matching removal and issue calculations.
        observation_proportions = missing.mean(axis=1)
        figure.add_trace(go.Bar(
            x=x,
            y=observation_proportions,
            marker_color="#154c79",
            hovertemplate="%{y:.1%} missing<extra></extra>",
            showlegend=False,
        ), row=2, col=1)
        figure.update_xaxes(showticklabels=False, range=[-0.5, len(x) - 0.5], row=1, col=1)
        figure.update_yaxes(tickmode="array", tickvals=y, ticktext=list(map(str, columns)), range=[len(y) - 0.5, -0.5], row=1, col=1)
        figure.update_yaxes(showticklabels=False, range=[len(y) - 0.5, -0.5], row=1, col=2)
        figure.update_xaxes(title_text="Variable missingness", tickformat=".0%", range=[0, 1], row=1, col=2)
        figure.update_xaxes(title_text="Observations", showticklabels=False, range=[-0.5, len(x) - 0.5], row=2, col=1)
        figure.update_yaxes(title_text="Missingness", tickformat=".0%", range=[0, 1], row=2, col=1)
        if show_thresholds:
            figure.add_vline(x=variable_threshold, line_dash="dash", line_color="#b22222", row=1, col=2)
            figure.add_hline(y=observation_threshold, line_dash="dash", line_color="#b22222", row=2, col=1)
    else:
        figure = go.Figure(heatmap)
        figure.update_xaxes(title_text="Observations", showticklabels=False, range=[-0.5, len(x) - 0.5])
        figure.update_yaxes(tickmode="array", tickvals=y, ticktext=list(map(str, columns)), range=[len(y) - 0.5, -0.5])

    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 15, "r": 20, "t": 10, "b": 35},
        showlegend=False,
        modebar={"orientation": "v"}
    )
    return figure


def instance():
    """Create the mutable missingness-map card."""
    this = Card(file=__file__, mutable=True)
    this.long_name = "Excessive missingness"
    this.description = "This card maps missing values across variables and observations and can remove excessive missingness from the dataset."

    def front():
        return ui.TagList(
            ui.span("Missingness map", class_="text-primary text-center d-block"),
            shinywidgets.output_widget(
                id="Map", fill=True, guide=this, title="Missingness map",
                text="Brown cells are missing values; pale cells are present values. Full screen also shows variable and observation missingness summaries.",
                position="left",
            ),
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.span("Missingness issues", class_="text-primary text-center d-block"),
            ui.output_ui(id="Table", guide=this, title="Missingness issues", text="Lists special-role variables with missing values and variables or observations above their thresholds.", position="left"),
        )

    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(id="Check"),
            ui.input_checkbox_group(
                id="Remove", label="Remove  excessive missingness in",
                choices=["Variables", "Observations"],
                inline=True, width="500px", guide=this,
                title="Remove excessive missingness", position="top",
                text="Removes variables or observations above their respective thresholds from this card's exported data. When both are selected, variables are removed first.",
            ),
            class_="vertically-scrollable-footer",
        )

    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_slider(
                id="VariableThreshold", label="Excessive variable missingness (%)", min=0, max=100, value=50, step=1,
                guide = this, text = "The excessive-missingness threshold of the <em>variables</em> as a percentage", position="left"
            ),
            ui.input_slider(
                id="ObservationThreshold", label="Excessive observation missingness (%)", min=0, max=100, value=50, step=1,
                guide = this, text = "The excessive-missingness threshold of the <em>observations</em> as a percentage", position="left"

            ),
            ui.input_checkbox(
                id="HideComplete", label="Hide variables without missing values", value=True,
                guide = this, text = "Whether to hide variables that have zero missing values", position="left"
            ),
            ui.input_checkbox(
                id="SortVariables", label="Sort variables by missingness", value=True,
                guide = this, text = "Sort the variables by descending degrees of missingness", position="left"
            ),
            ui.input_checkbox(
                id="ShowThresholds", label="Show thresholds in full screen", value=True,
                guide = this, text = "Display the two thresholds on the chart (when in full-screen mode)", position="left"
            ),
            ui.input_slider(
                id="MaxObs", label="Maximum observations to analyse", min=3, max=7, value=4, ticks=True, pre="10^",
                guide=this, text = 'Limit to number of observations to analyse to ensure responsiveness (logarithmic scale).', position="left",
            ),        
        )

    this.settings = settings

    def server(input, output, session):
        @this.suspendable(calc=True)
        def incomingproxy_data():
            return this.input_data()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def Remove():
            return input.Remove() or []

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def VariableThreshold():
            return float(input.VariableThreshold()) / 100

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def ObservationThreshold():
            return float(input.ObservationThreshold()) / 100

        @this.suspendable(calc=True)
        @this.record_code
        def TransformedData():
            source = incomingproxy_data()
            selected = Remove()
            return _transform_data(
                source,
                remove_variables="Variables" in selected,
                remove_observations="Observations" in selected,
                variable_threshold=VariableThreshold(),
                observation_threshold=ObservationThreshold(),
            )

        @this.suspendable(calc=True)
        def DisplayFrame():
            return _display_frame(TransformedData().frame, 10 ** int(input.MaxObs()))

        @output
        @render_widget
        def Map():
            full_screen = bool(this.isFullScreen())
            figure = _missingness_figure(
                DisplayFrame(),
                hide_complete=bool(input.HideComplete()),
                sort_variables=bool(input.SortVariables()),
                variable_threshold=VariableThreshold(),
                observation_threshold=ObservationThreshold(),
                show_thresholds=bool(input.ShowThresholds()),
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
        def Table():
            return ui.output_data_frame(id="Table2")

        @output
        @render.data_frame
        def Table2():
            proxy = TransformedData()
            return render.DataTable(
                _issues_table(proxy.frame, proxy.role_map, VariableThreshold(), ObservationThreshold()),
                width="100%", height="98%",
            )

        @output
        @render.ui
        def Check():
            source = incomingproxy_data()
            original = source.frame
            transformed = TransformedData().frame
            selected = Remove()
            after_variables = _remove_excessive_variables(source, VariableThreshold())
            removed_variables = original.shape[1] - transformed.shape[1]
            removed_observations = original.shape[0] - transformed.shape[0]
            excessive_variables = original.shape[1] - after_variables.frame.shape[1]
            observation_basis = (
                after_variables.frame
                if "Variables" in selected
                else original
            )
            excessive_observations = int(_excessive_observation_mask(
                observation_basis, ObservationThreshold()
            ).sum())
            messages = []
            if "Variables" in selected:
                messages.append(f"Removed {removed_variables} excessively missing variables")
            elif excessive_variables:
                messages.append(f"{excessive_variables} variables exceed their threshold")
            if "Observations" in selected:
                messages.append(f"removed {removed_observations} excessively missing observations")
            elif excessive_observations:
                messages.append(f"{excessive_observations} observations exceed their threshold")
            if not messages:
                return ui.span("No variables or observations exceed their thresholds.", class_="text-success")
            return ui.span("; ".join(messages), class_="text-warning")

        return TransformedData

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
