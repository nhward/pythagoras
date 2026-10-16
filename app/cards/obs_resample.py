"""Evaluation resampling card: metadata only, with compact membership previews."""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from dataclasses import replace
from html import escape
from pathlib import Path

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
from evaluation_plan import (
    GOALS,
    OBJECTIVES,
    PURPOSES,
    STATES,
    STRATEGIES,
    build_plan,
    evaluation_frame,
    goal_options,
    level_key,
    partition_levels,
    role_columns,
    stratifier_choices,
)
from module import Module
from proxy_data import proxy_data
from roles import Role, RoleMap
from selection_restore import SelectionRestore
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget

COLORS = ("#e6e9ed", "#427ba6", "#e8a13d", "#8755a1", "#309b8b")


#TODO: Maybe make flip-side show the allocations tabulated (as per the current download). The current table seems rather pointless.
#TODO: Drop the download button.

def _mapping_id(column, key):
    return "Partition_" + hashlib.sha256(f"{column}\0{key}".encode()).hexdigest()[:16]


def _unavailable(source, **options):
    result = {}
    for goal in GOALS:
        if goal == "none":
            continue
        try:
            goal_options(source, goal, **options)
        except (ValueError, TypeError) as error:
            result[goal] = str(error)
    return result


@recordable
def _overview(source, plan, *, columns=600, rows=60):
    """Bound both chart dimensions, retaining exact counts for aggregated bins."""
    if not plan.ready:
        return None
    r = plan.recipe
    frame = evaluation_frame(source)
    n = plan.size
    order = np.arange(n)
    ordering = "Source order"
    if r["objective"] in {"future", "known"}:
        order = np.argsort(frame[r["sequence"]].to_numpy(), kind="stable")
        ordering = "Sequence order"
    elif r.get("stratifier"):
        codes, _ = pd.factorize(frame[r["stratifier"]], sort=False)
        order = np.argsort(codes, kind="stable")
        ordering = "Stratifier order"
    elif r["strategy"] in {"grouped", "logo"}:
        codes, _ = pd.factorize(frame[r["group"]], sort=False)
        order = np.argsort(codes, kind="stable")
        ordering = "Group order"
    nx, ny = min(columns, n), min(rows, plan.count)
    xbin = np.arange(n) * nx // n
    counts = np.zeros((ny, nx, 5), dtype=np.int64)
    for j, (train, validation) in enumerate(plan._splits()):
        state = plan.membership(train, validation)[order]
        np.add.at(counts[j * ny // plan.count], (xbin, state), 1)
    strata = None
    if r.get("stratifier"):
        codes, levels = pd.factorize(frame[r["stratifier"]], sort=False)
        strata_counts = np.zeros((nx, len(levels)), dtype=np.int64)
        np.add.at(strata_counts, (xbin, codes[order]), 1)
        strata = {"name": r["stratifier"], "labels": list(map(str, levels)), "counts": strata_counts}
    details = []
    xedges = np.ceil(np.arange(nx + 1) * n / nx).astype(int)
    yedges = np.ceil(np.arange(ny + 1) * plan.count / ny).astype(int)
    for x in range(nx):
        positions = order[xedges[x]:xedges[x + 1]]
        if len(positions) == 1:
            p = int(positions[0])
            detail = f"Source position {p}"
            for key in ("group", "sequence", "entity", "stratifier"):
                if r.get(key):
                    detail += f"<br>{escape(r[key])}: {escape(str(frame[r[key]].iloc[p]))}"
        else:
            detail = f"Display observations {xedges[x] + 1}–{xedges[x + 1]}"
        details.append(detail)
    return {"counts": counts, "details": details, "order": ordering, "n": n, "splits": plan.count, "yedges": yedges, "strata": strata, "positions": order}


def _figure(overview, *, full=False):
    if overview is None:
        return Card.empty_figure("Choose a prediction goal and complete its settings.")
    counts = overview["counts"]
    yedges = overview["yedges"]
    strata = overview["strata"]
    strata_counts = strata["counts"] if strata else None
    # Both dimensions aggregate contiguous bins; all memberships still exported.
    if not full:
        old_y, old_x = counts.shape[:2]
        ny, nx = min(25, old_y), min(1000, old_x)
        reduced = np.zeros((ny, nx, 5), dtype=np.int64)
        for y in range(old_y):
            np.add.at(reduced[y * ny // old_y], np.arange(old_x) * nx // old_x, counts[y])
        counts = reduced
        if strata is not None:
            strata_counts = np.zeros((nx, len(strata["labels"])), dtype=np.int64)
            np.add.at(strata_counts, np.arange(old_x) * nx // old_x, strata["counts"])
        yedges = yedges[np.ceil(np.arange(ny + 1) * old_y / ny).astype(int)]
    ny, nx = counts.shape[:2]
    # Most frequent state wins; argmax breaks ties in the fixed legend order.
    z = counts.argmax(axis=2)
    aggregate = nx < overview["n"] or ny < overview["splits"]
    hover = []
    for y in range(ny):
        line = []
        for x in range(nx):
            detail = overview["details"][x] if full else "Open fullscreen for source positions and ranges"
            line.append(detail + "<br>" + "<br>".join(f"{STATES[k]}: {counts[y, x, k]:,}" for k in range(5) if counts[y, x, k]))
        hover.append(line)
    labels = []
    for y in range(ny):
        start = int(yedges[y]) + 1
        end = int(yedges[y + 1])
        labels.append(f"CV{start}" if start == end else f"CV{start}–{end}")
    scale = [[v, color] for i, color in enumerate(COLORS) for v in (i / len(COLORS), (i + 1) / len(COLORS))]
    figure = go.Figure(go.Heatmap(z=z, x=np.arange(1, nx + 1), y=labels, zmin=-.5, zmax=len(COLORS)-.5,
        colorscale=scale, showscale=False, text=hover, hovertemplate="%{y}<br>%{text}<extra></extra>", ygap=1))
    for k in np.flatnonzero(counts.sum(axis=(0, 1))):
        figure.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker={"color": COLORS[k], "symbol": "square", "size": 9}, name=STATES[k], hoverinfo="skip"))
    if strata is not None:
        palette = ("#7b3294", "#008837", "#c51b7d", "#a6611a", "#1b9e77", "#7570b3", "#b2182b", "#80cdc1")
        labels_strata = strata["labels"]
        colors = [palette[i % len(palette)] for i in range(len(labels_strata))]
        bins = strata_counts.argmax(axis=1)
        scale_strata = [[v, color] for i, color in enumerate(colors)
                        for v in (i / len(colors), (i + 1) / len(colors))]
        text = ["<br>".join(f"{escape(label)}: {counts_:,}" for label, counts_ in zip(labels_strata, row) if counts_)
                for row in strata_counts]
        figure.add_trace(go.Heatmap(x=np.arange(1, nx + 1), y=["Strata"], z=[bins], yaxis="y2",
            zmin=-.5, zmax=len(colors)-.5, colorscale=scale_strata, showscale=False,
            text=[text], hovertemplate=escape(strata["name"])+"<br>%{text}<extra></extra>"))
        figure.update_layout(yaxis={"domain": [.16, 1]}, yaxis2={"domain": [0, .065], "type": "category",
            "tickfont": {"size": 10}, "fixedrange": True, "showgrid": False})
        if full:
            for i, label in enumerate(labels_strata):
                figure.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                    marker={"color": colors[i], "symbol": "square", "size": 9}, legendgroup="strata",
                    legendgrouptitle_text=f"Stratifier: {strata['name']}", hoverinfo="skip"))
            if overview["order"] == "Stratifier order":
                for edge in np.flatnonzero(bins[1:] != bins[:-1]) + 1:
                    figure.add_shape(type="line", x0=edge+.5, x1=edge+.5, y0=0, y1=1,
                                     xref="x", yref="paper", line={"color": "#777", "width": .5}, opacity=.3)
    figure.update_layout(template="plotly_white", margin={"l": 52 if full else 40, "r": 8, "t": 4, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"size": 12 if full else 10},
        xaxis={"title": {"text": overview["order"] + (" · aggregated bins" if aggregate else " · observation"), "standoff": 10},
               "anchor": "y2" if strata is not None else "y", "showticklabels": full},
        yaxis={"autorange": "reversed", "type": "category", "nticks": 12 if full else 6},
        legend={"orientation": "h", "y": -.2, "x": .5, "xanchor": "center", "font": {"size": 11 if full else 9}},
        hovermode="closest" if full else False, showlegend=True)
    return figure


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = "Observation resampling"
    this.description = "Choose a prediction goal; assigned roles determine evaluation splits."
    this.defer_configuration_input("Goal")
    this.defer_configuration_input("Stratifier")
    this.register_configuration_state_field("evaluation_plan")
    this.register_configuration_state_field("partition_mapping")
    this.front = lambda: ui.TagList(
        ui.tags.style(
            f"#{this.ns('CardBody')}:not(.flipped) > .flippable > .back {{ visibility: hidden; }}"
            f"#{this.ns('CardBody')}.flipped > .flippable > .front {{ visibility: hidden; }}"
        ),
        ui.output_ui("Summary"),
        shinywidgets.output_widget("Chart", fill=True, guide=this, title="Evaluation membership", position="left",
            text="Splits show training, validation and unused rows. The separate Stratifier strip identifies strata. Fullscreen adds category legends and source-position hover; aggregated bins show the most frequent state, with ties resolved in legend order."),
    )
    this.back = lambda: ui.TagList(
        ui.output_ui("Details"),
        ui.output_data_frame("Table", guide=this, title="Split checks", position="left",
            text="Exact counts, group overlap, time ranges and target-class checks for every split."),
        ui.download_button("Membership", label="Download membership", class_="btn-sm",
            guide=this, position="top", text="One row per source observation per split. Zero-based source positions remain unchanged by chart sorting."),
    )
    this.footer = lambda: ui.TagList(
        ui.output_ui(id="GoalControl"), 
        ui.output_ui(id="Status")
    )

    def number(name, label, value, minimum, text):
        return ui.input_numeric(name, label=label, value=value, min=minimum, step=1,
                                guide=this, position="left", text=text)

    this.settings = lambda: ui.TagList(
        ui.output_ui("Roles"),
        ui.input_select("Stratifier", label="Stratifier", choices={"": "None"}, selected="",
            guide=this, position="left",
            text="Choose an assigned Stratifier to preserve proportions or withhold whole strata. Cluster memberships appear last. None disables stratification; New strata requires a selection."),
        ui.output_ui("PartitionMapping"),
        number("Folds", "Number of folds (K)", 5, 2,
               "For generated New observations, New strata and New entities splits. For grouped goals, use the smaller of K and the available development groups; at least two groups are required."),
        number("Seed", "Random seed", 42, 0,
               "For generated observation and group folds. Randomly assign whole strata/entities to folds reproducibly. Not used by supplied validation or temporal windows."),
        number("Window", "Initial training window (distinct values)", None, 1, "For generated Future observations only: size of the first training window."),
        number("Horizon", "Validation window (distinct values)", None, 1, "For generated Future observations only: forecast window in distinct Sequence values. Ties stay together."),
        number("Gap", "Gap (distinct values)", 0, 0, "For generated Future observations only: unused values between training and validation."),
        number("Step", "Step (blank = validation window)", None, 1, "For generated Future observations only: advance by at least the validation-window size."),
        ui.input_checkbox("Expanding", label="Expand training window", value=True, guide=this, position="left",
            text="For generated Future observations only: retain earlier training values; unchecked keeps a fixed width. All settings remain available even when irrelevant to the active goal."),
    )

    def server(input, output, session):
        busy = this.busy()
        saved_stratifier = this.restored_configuration_input("Stratifier")
        selection = SelectionRestore(None if saved_stratifier is None else ([saved_stratifier] if saved_stratifier else []))

        @reactive.effect
        def ObserveStratifier():
            value = input.Stratifier()
            selection.observe([value] if value else [])

        saved_goal = this.restored_configuration_input("Goal") or "none"
        restored = this.restored_configuration_field("evaluation_plan")
        saved_mapping = this.restored_configuration_field("partition_mapping", {}) or {}
        mapping_cache = dict(saved_mapping.get("levels", {}))
        mapping_variable = saved_mapping.get("variable")

        @this.reactable(calc=True)
        def Incoming():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(source is not None)
            return source

        @this.reactable()
        def StratifierChoices():
            choices = stratifier_choices(Incoming())
            with reactive.isolate():
                current = input.Stratifier()
            selected = selection.resolve([current] if current else [], choices, choices[:1])
            if current and current not in choices and selection.pending is None:
                selected = choices[:1]
            ui.update_select("Stratifier", choices={**{name: name for name in choices}, "": "None"},
                             selected=selected[0] if selected else "")

        @this.reactable(calc=True)
        def Partition():
            source = Incoming()
            columns = role_columns(source, Role.PARTITION)
            if len(columns) != 1:
                return None, {}
            column = columns[0]
            return column, partition_levels(source.clean_frame[column])

        @this.reactable(calc=True)
        def Mapping():
            column, levels = Partition()
            result = {}
            for key in levels:
                name = _mapping_id(column, key)
                try:
                    value = input[name]()
                except SilentException:
                    value = mapping_cache.get(key, "") if column == mapping_variable else ""
                result[key] = value or ""
            return result

        @this.reactable()
        def RememberMapping():
            nonlocal mapping_variable, mapping_cache
            values = Mapping()
            mapping_variable = Partition()[0]
            mapping_cache = values.copy()

        # Downstream data needs these inputs even in an unopened upstream section.
        @output(suspend_when_hidden=False)
        @render.ui
        def PartitionMapping():
            column, levels = Partition()
            if column is None:
                return ui.tags.small("No Partition assigned; validation is generated from the footer goal.")
            with reactive.isolate():
                current = Mapping()
            counts = Incoming().clean_frame[column].map(level_key).value_counts()
            rows = []
            for key, label in levels.items():
                name = _mapping_id(column, key)
                this.exclude_configuration_input(name)
                rows.append(ui.tags.tr(
                    ui.tags.td(label), ui.tags.td(f"{counts[key]:,}"),
                    ui.tags.td(ui.input_select(name, label=None, choices=PURPOSES, selected=current.get(key, ""),
                        width="100%", guide=this, title=f"Partition: {label}", position="left",
                        text="Assign this level to Development, Validation, Final holdout or Prediction. Every level needs an assignment. Multiple levels may share a purpose."))))
            return ui.TagList(ui.tags.strong(f"Partition: {column}"),
                ui.tags.table(ui.tags.thead(ui.tags.tr(*(ui.tags.th(x) for x in ("Level", "Rows", "Purpose")))),
                              ui.tags.tbody(*rows), class_="table table-sm"),
                ui.tags.small("Validation rows define the split. Otherwise the footer goal splits development data. Holdout and prediction rows remain outside tuning."))

        # Downstream data needs these inputs even in an unopened upstream section.
        @output(suspend_when_hidden=False)
        @render.ui
        def GoalControl():
            with reactive.isolate():
                try:
                    selected = input.Goal() or saved_goal
                except SilentException:
                    selected = saved_goal
            control = ui.input_radio_buttons("Goal", label="Predicting", choices=GOALS, selected=selected, inline=True,
                guide=this, title="Prediction goal", position="top",
                text="Choose the predictions to evaluate. Assigned roles determine the method: Stratifier preserves proportions or defines withheld strata; longitudinal Identifier/Sequence defines entities and future known-entity predictions. None passes incoming metadata through.")
            return control

        @output
        @render.ui
        def Roles():
            source = Incoming()
            descriptions = [f"{r.value.title()}: {', '.join(role_columns(source, r))}" for r in
                (Role.SEQUENCE, Role.IDENTIFIER, Role.STRATIFIER, Role.TARGET) if role_columns(source, r)]
            return ui.div(ui.tags.small(" · ".join(descriptions) or "No split roles assigned."),
                *(ui.div(ui.tags.small(f"{GOALS[k]}: {v}")) for k, v in _unavailable(source, stratifier=input.Stratifier() or None).items()))

        @this.reactable(calc=True)
        def CurrentOptions():
            def integer(name):
                value = input[name]()
                return int(value) if value is not None and float(value).is_integer() else value
            goal = input.Goal() or "none"
            options = {"goal": goal}
            if goal == "none":
                return options
            options["stratifier"] = input.Stratifier() or None
            options["partition_map"] = Mapping()
            if "validation" not in options["partition_map"].values():
                if goal == "future":
                    options.update(window=integer("Window"), horizon=integer("Horizon"), gap=integer("Gap"),
                        step=integer("Step") if input.Step() is not None else integer("Horizon"), expanding=bool(input.Expanding()))
                else:
                    options.update(folds=integer("Folds"), seed=integer("Seed"))
            return options

        @this.reactable(calc=True)
        @this.settle(seconds=2, bypass_during_tests=False)
        def Options():
            return CurrentOptions()

        @busy.track("Checking evaluation splits…")
        @this.extended_task
        @this.record_context
        async def Calculate(source, options):
            def work():
                plan = build_plan(source, **options)
                if restored:
                    if restored.get("fingerprint") == plan.fingerprint and restored.get("recipe") == plan.recipe:
                        plan = replace(plan, holdout_used=bool(restored.get("holdout_used", False)))
                    elif restored.get("fingerprint") != plan.fingerprint:
                        plan = replace(plan, warnings=(*plan.warnings, "Plan regenerated for current source."))
                return plan, _overview(source, plan)
            result = work() if Module.IS_SHINYLIVE else await asyncio.to_thread(work)
            return source, options, result

        @this.reactable()
        def Start():
            Calculate.cancel()
            options = Options()
            if options["goal"] != "none":
                Calculate.invoke(Incoming().clone(), options)

        @this.reactable(calc=True)
        def Results():
            req(Options()["goal"] != "none")
            try:
                source, options, result = Calculate.result()
            except SilentOperationInProgressException:
                req(False)
            req(options == Options() and source.equals(Incoming()))
            return result

        @this.reactable(calc=True)
        @this.record_context
        def Export():
            source = Incoming()
            if Options()["goal"] == "none":
                return source
            return source.with_evaluation_plan(Results()[0], card=this.name)

        def state():
            if Options()["goal"] == "none":
                return None
            return Results()[0].bookmark()
        this.register_configuration_state_provider("evaluation_plan", state)
        this.register_configuration_state_provider("partition_mapping", lambda: {"variable": Partition()[0], "levels": Mapping()})

        @output
        @render.ui
        def Summary():
            if Options()["goal"] == "none":
                return ui.div("Choose a prediction goal", class_="text-primary text-center small")
            try:
                plan = Results()[0]
            except SilentException:
                return ui.div(busy.ui(), class_="small")
            if not plan.ready:
                return ui.div("Plan needs attention", class_="text-danger text-center small")
            r = plan.recipe
            text = f"{plan.count:,} {'split' if plan.count == 1 else 'splits'} · {len(plan.development):,} development"
            detail = STRATEGIES[r["strategy"]]
            if this.isFullScreen():
                detail += " · " + OBJECTIVES[r["objective"]]
            return ui.div(ui.div(text), ui.tags.small(detail), class_="text-primary text-center")

        @output
        @render.ui
        def Status():
            if Options()["goal"] == "none":
                return ui.span("Incoming evaluation metadata passes through unchanged.", class_="text-secondary")
            try:
                plan = Results()[0]
            except SilentException:
                return ui.span("Waiting for settings to settle or recalculating…", class_="text-primary")
            if plan.errors:
                return ui.span(plan.errors[0], class_="text-danger", role="alert")
            messages = [f"{len(getattr(plan, key)):,} {label}" for key, label in
                        (("validation", "fixed validation"), ("holdout", "holdout"), ("prediction", "prediction")) if getattr(plan, key)]
            adjustments = [note for note in plan.warnings if note.startswith("Using K=")]
            messages.extend(adjustments)
            if "Plan regenerated for current source." in plan.warnings:
                messages.append("Plan regenerated for current source.")
            if not plan.supervised:
                messages.append("Target checks pending")
            if any("dependence" in note for note in plan.warnings):
                messages.append("Dependence warning — see flip side")
            if any("omit target" in note for note in plan.warnings):
                messages.append("Class warning — see flip side")
            if plan.holdout_used:
                messages.append("Final holdout already used")
            if Incoming().evaluation_plan:
                messages.append("Replaces incoming plan")
            warning = not plan.supervised or plan.holdout_used or any("warning" in m.lower() for m in messages)
            color = "warning" if warning else "primary" if messages else "success"
            return ui.span(" · ".join(messages) or "Ready · supervised checks complete", class_=f"text-{color}")

        @output
        @render_widget
        def Chart():
            full = bool(this.isFullScreen())
            try:
                figure = _figure(Results()[1], full=full)
            except SilentException:
                figure = Card.empty_figure("Choose a prediction goal and complete its settings.")
            widget = go.FigureWidget(figure)
            widget._config = {"displayModeBar": full, "displaylogo": False, "responsive": True}
            return widget

        @output
        @render.ui
        def Details():
            plan = Results()[0]
            return ui.TagList(ui.tags.strong("Split diagnostics"), *(ui.div(note, class_="small") for note in (*plan.errors, *plan.warnings)),
                ui.div(f"Fixed validation: {len(plan.validation):,} · Final holdout: {len(plan.holdout):,} · Prediction: {len(plan.prediction):,}", class_="small"))

        @output
        @render.data_frame
        def Table():
            return render.DataTable(Results()[0].diagnostics, width="100%", height=None)

        @output
        @render.download(filename="evaluation-membership.csv")
        def Membership():
            for j, frame in enumerate(Results()[0].membership_frames(Incoming())):
                yield frame.to_csv(index=False, header=j == 0)

        session.on_ended(Calculate.cancel)
        return Export

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    frame = pd.DataFrame({"value": np.arange(60, dtype=float), "time": np.repeat(np.arange(20), 3),
                          "entity": np.tile(["A", "B", "C"], 20), "site": pd.Categorical(np.tile(["North", "South", "West"], 20)),
                          "class": pd.Categorical(np.tile(["yes", "no"], 30))})
    this._imports.set(proxy_data(_df=frame, _roles=RoleMap.from_primitive({"predictor": ["value"],
        "sequence": ["time"], "identifier": ["entity"], "stratifier": ["site"], "target": ["class"]})))
    this.run()
