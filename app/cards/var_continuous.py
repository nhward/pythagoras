"""Inspect decimal variables through sorted values and cumulative ranks."""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from code_recording import recordable
from module import Module
from plotly.colors import qualitative
from proxy_data import proxy_data
from roles import Role
from selection_restore import SelectionRestore
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget
from var_types import var_kind

DEFAULT_VARIABLES = 15
SAMPLE_SEED = 2025
MIN_DISTINCT_FRACTION = .90
SUMMARY_COLUMNS = [
    "Variable", "Finite values", "Omitted", "Distinct values", "Distinct %",
    "Gap from", "Gap to", "Largest gap", "Gap / range %", "Summary",
]


@recordable
def _eligible_columns(data, use_target=False):
    """Decimal targets (when enabled), treatments, then predictors."""
    excluded = {Role.IDENTIFIER, Role.WEIGHTING}
    allowed = {Role.PREDICTOR, Role.TREATMENT}
    if use_target:
        allowed.add(Role.TARGET)
    columns = [column for column in data.columns
               if data.role_map.roles_for(column) & allowed
               and not data.role_map.roles_for(column) & excluded
               and (use_target or Role.TARGET not in data.role_map.roles_for(column))
               and not str(column).startswith(Card.SHADOW_PREFIX)
               and var_kind(data.frame[column]) == "decimal"]
    return sorted(columns, key=lambda column:
                  0 if Role.TARGET in data.role_map.roles_for(column) else
                  1 if Role.TREATMENT in data.role_map.roles_for(column) else 2)



@recordable
@dataclass
class ContinuousAnalysis:
    curves: dict
    colors: dict
    table: pd.DataFrame
    sampled: int
    total: int
    message: str = ""


@recordable
def _gap_summary(name, values, sampled):
    """Summarize one sorted finite sample without testing for significance."""
    count = len(values)
    distinct = 1 + int(np.count_nonzero(values[1:] != values[:-1])) if count else 0
    fraction = distinct / count if count else 0.
    row = dict(zip(SUMMARY_COLUMNS, [
        str(name), count, sampled - count, distinct, 100 * fraction if count else np.nan,
        np.nan, np.nan, np.nan, np.nan, "",
    ]))
    if count < 2:
        row["Summary"] = "Too few finite values" if count else "No finite values"
    elif fraction < MIN_DISTINCT_FRACTION:
        row["Summary"] = "Withheld: fewer than 90% distinct values"
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            gaps = np.diff(values)
            span = values[-1] - values[0]
        if not np.isfinite(span) or not np.isfinite(gaps).all():
            row["Summary"] = "Gap exceeds floating-point range"
        elif span > 0:
            index = int(np.argmax(gaps))
            row.update({
                "Gap from": values[index], "Gap to": values[index + 1],
                "Largest gap": gaps[index], "Gap / range %": 100 * (gaps[index] / span),
                "Summary": "Descriptive only; not a significance test",
            })
    return row


@recordable
def _analyze(data, variables, limit=10_000, cancel=None, use_target=False):
    """Sample rows before sorting; retain each variable's finite observations."""
    eligible = _eligible_columns(data, use_target)
    selected = [column for column in eligible if column in variables]
    total = len(data.frame)
    limit = max(1, int(limit))
    positions = (np.sort(np.random.default_rng(SAMPLE_SEED).choice(total, limit, replace=False))
                 if total > limit else np.arange(total))
    colors = {column: qualitative.Dark24[index % len(qualitative.Dark24)]
              for index, column in enumerate(eligible)}
    curves, rows = {}, []
    for column in selected:
        if cancel is not None and cancel.is_set():
            return ContinuousAnalysis({}, {}, pd.DataFrame(columns=SUMMARY_COLUMNS),
                                      len(positions), total, "Calculation superseded.")
        values = data.frame[column].iloc[positions].to_numpy(dtype=float, na_value=np.nan)
        values = np.sort(values[np.isfinite(values)], kind="stable")
        curves[column] = values
        rows.append(_gap_summary(column, values, len(positions)))
    message = ""
    if not eligible:
        message = "No eligible decimal variables."
    elif not selected:
        message = "Select at least one decimal variable."
    elif not any(len(values) for values in curves.values()):
        message = "No finite decimal values in the assessed rows."
    return ContinuousAnalysis(curves, colors, pd.DataFrame(rows, columns=SUMMARY_COLUMNS),
                              len(positions), total, message)


@recordable
def _figure(result, full_screen=False):
    if result.message:
        return Card.empty_figure(result.message)
    figure = go.Figure()
    for name, values in result.curves.items():
        count = len(values)
        if not count:
            continue
        ranks = np.arange(1, count + 1)
        with np.errstate(over="ignore"):
            span = values[-1] - values[0]
        if span == 0:
            scaled = np.zeros(count)
        elif np.isfinite(span):
            scaled = (values - values[0]) / span
        else:
            # Equivalent scaling without overflowing for extreme finite values.
            half = values / 2
            scaled = (half - half[0]) / (half[-1] - half[0])
        figure.add_trace(go.Scattergl(
            x=100 * (ranks / count), y=scaled, customdata=np.column_stack((values, ranks)),
            name=str(name), uid=str(name), mode="lines" if count > 1 else "markers",
            line={"color": result.colors[name], "width": 2},
            marker={"color": result.colors[name], "size": 7},
            hovertemplate=("Value: %{customdata[0]:.6g}<br>Cumulative rank: %{x:.3g}%"
                           f"<br>Position: %{{customdata[1]:.0f}} of {count:,}"
                           "<extra>%{fullData.name}</extra>"),
        ))
    figure.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#bbd6f8",
        yaxis={"title": "Range-standardized value", "range": [0, 1],
               "showticklabels": False, "ticks": "", "fixedrange": not full_screen},
        xaxis={"title": "Cumulative rank (%)", "range": [0, 100], "fixedrange": not full_screen},
        showlegend=full_screen,
        legend={"orientation": "h", "y": 1.02, "x": .5, "xanchor": "center", "yanchor": "bottom"},
        margin={"l": 55, "r": 15, "t": 10, "b": 45},
        modebar={"orientation": "v"}, font={"size": 13 if full_screen else 10},
        uirevision="continuous-values",
    )
    return figure


def instance():
    this = Card(file=__file__, mutable=False)
    this.long_name = "Variable continuity"
    this.description = "Inspect sorted decimal variables for discontinuities."
    this.defer_configuration_input("Variables")
    this.front = lambda: ui.TagList(
        ui.span("Sorted values and cumulative ranks", class_="text-primary text-center d-block"),
        shinywidgets.output_widget(
            id="Chart", fill=True, guide=this, title="Continuous values", position="left",
            text="Each line plots a decimal variable's sorted finite values against cumulative rank. Steep stretches span gaps; horizontal stretches indicate repeated values. Hover identifies the variable and rank. Full screen adds a clickable legend and zoom. The vertical axis scales each variable to (x-min)/(max-min); hover retains original values. Constant variables are placed at zero.",
        ),
    )
    this.back = lambda: ui.TagList(
        ui.p("Largest observed gaps, not significance tests. Gap summaries are withheld below 90% distinct values. Counts and percentages describe the assessed sample."),
        ui.output_data_frame(
            id="Summary", guide=this, title="Gap summary", position="left",
            text="One row per selected variable. Omitted counts missing or infinite values in the sampled rows. Gap from/to bound the largest adjacent gap, and Gap / range expresses its size relative to that variable's observed range. Equal largest gaps report the first. Sampling can exaggerate gaps.",
        ),
    )
    this.footer = lambda: ui.TagList(ui.output_ui("Busy"), ui.output_text("Status"))
    this.settings = lambda: ui.TagList(
        ui.input_checkbox(
            id="UseTarget", label="Include any target", value=False,
            guide=this, position="left",
            text="Allow decimal Target-role variables in the picker. Decimal Treatment-role variables are always eligible.",
        ),
        ui.input_selectize(
            id="Variables", label="Variables", choices=[], selected=[], multiple=True,
            options={"plugins": ["remove_button"]}, guide=this, position="left",
            text="Choose decimal predictors and treatments, plus targets when 'Include any target' is enabled. The first 15 eligible variables are selected initially.",
        ),
        ui.input_slider(
            id="Limit", label="Limit (observations)", min=2, max=5, value=4, step=1, pre="10^",
            guide=this, position="left",
            text="Cap the number of rows inspected before any sorting: 10^4 means 10,000 rows. Larger datasets use one reproducible random sample shared across variables. Missing and infinite values are then omitted separately per variable. Sampling can miss values and enlarge apparent gaps; increase the limit to investigate.",
        ),
    )

    def server(input, output, session):
        selection = SelectionRestore(this.restored_configuration_input("Variables"))
        busy = this.busy()
        cancellation = Event()

        @reactive.effect
        def ObserveSelection():
            selection.observe(input.Variables() or [])

        @this.reactable(calc=True)
        def Incoming():
            try:
                data = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(data is not None)
            return data

        @this.reactable()
        @this.record_context
        def UpdateChoices():
            eligible = _eligible_columns(Incoming(), input.UseTarget())
            with reactive.isolate():
                current = list(input.Variables() or [])
            selected = selection.resolve(current, eligible, eligible[:DEFAULT_VARIABLES])
            ui.update_selectize("Variables", choices=eligible, selected=selected)

        @this.reactable(calc=True)
        @this.settle(seconds=2)
        def Options():
            return {"variables": tuple(input.Variables() or ()), "limit": int(10**input.Limit()),
                    "use_target": bool(input.UseTarget())}

        @busy.track("Sorting continuous values…")
        @this.extended_task
        @this.record_context
        async def Calculate(data, options, cancelled):
            if Module.IS_SHINYLIVE:
                result = _analyze(data, **options, cancel=cancelled)
            else:
                result = await asyncio.to_thread(_analyze, data, **options, cancel=cancelled)
            return data, options, result

        @this.reactable()
        def Start():
            nonlocal cancellation
            source, options = Incoming(), Options()
            cancellation.set()
            Calculate.cancel()
            cancellation = Event()
            Calculate.invoke(source, options, cancellation)

        @this.reactable(calc=True)
        def Results():
            try:
                source, options, result = Calculate.result()
            except SilentOperationInProgressException:
                req(False)
            current = Incoming()
            req((source is current or source.equals(current)) and options == Options())
            return result

        @output
        @render_widget
        @this.record_context
        def Chart():
            full = bool(this.isFullScreen())
            try:
                figure = _figure(Results(), full_screen=full)
            except SilentException:
                figure = Card.empty_figure("Waiting for data or calculation.")
            widget = go.FigureWidget(figure)
            widget._config = {"displayModeBar": full, "displaylogo": False, "responsive": True}
            return widget

        @output
        @render.data_frame
        def Summary():
            table = Results().table.copy()
            for column in table.select_dtypes(include="floating"):
                table[column] = table[column].map(lambda value: f"{value:.4g}" if pd.notna(value) else "")
            return render.DataTable(table, width="100%", height=None, filters=False)

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render.text
        def Status():
            result = Results()
            if result.message:
                return result.message
            count = sum(bool(len(values)) for values in result.curves.values())
            noun = "variable" if count == 1 else "variables"
            text = f"{count} {noun} plotted; {result.sampled:,} of {result.total:,} rows assessed."
            if result.sampled < result.total:
                text += " Reproducible sample: apparent gaps may be enlarged."
            if result.table["Omitted"].sum():
                text += " Missing/nonfinite values omitted separately per variable."
            return text

        def stop():
            cancellation.set()
            Calculate.cancel()

        session.on_ended(stop)
        return Incoming

    this.server = server
    return this


if Module.running_directly(name=__name__):
    frame = pd.DataFrame({"gapped": np.r_[np.linspace(0, 1, 50), np.linspace(3, 4, 50)],
                          "smooth": np.linspace(0, 4, 100)})
    this = instance()
    this._imports.set(proxy_data(_df=frame, _name="Continuous values"))
    this.run()
