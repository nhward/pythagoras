"""Preview target balancing and register an unfitted training-only sampler."""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
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
from module import Module
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from proxy_data import proxy_data
from roles import Role, RoleMap
from scipy.stats import binom, chisquare
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget
from TargetBalanceSampler import TargetBalanceSampler, balance_weights
from var_types import var_kind


@recordable
def _eligibility(source):
    targets = [c for c in source.columns if Role.TARGET in source.role_map.roles_for(c)]
    if len(targets) != 1:
        return None, "Choose exactly one nominal Target upstream; balancing is not needed here."
    target = targets[0]
    if var_kind(source.frame[target]) != "nominal":
        return None, "Target balancing requires a nominal Target; this Target is not nominal."
    if source.frame[target].nunique() < 2:
        return None, "Target balancing needs at least two observed target classes."
    return target, ""


@recordable
def _diagnostic(frame, target, weight=None):
    weights = balance_weights(frame, weight)
    valid = frame[target].notna().to_numpy()
    labels = frame[target].iloc[np.flatnonzero(valid)]
    labels = labels.cat.remove_unused_categories()
    levels = list(labels.cat.categories)
    weights = weights[valid]
    totals = np.bincount(labels.cat.codes, weights=weights, minlength=len(levels))
    if len(levels) < 2 or not len(weights) or weights.sum() <= 0:
        raise ValueError("The equal-frequency diagnostic requires two classes with positive total mass.")
    # Kish effective size is invariant to changing the units of importance
    # weights. Weighted results remain approximate, not a survey-design test.
    scaled = weights / weights.max()
    effective = scaled.sum() ** 2 / np.square(scaled).sum()
    observed = totals / totals.sum() * effective
    expected = effective / len(levels)
    statistic = np.square(observed - expected).sum() / expected
    if statistic < 1e-12:
        pvalue, method = 1., "equal observed totals"
    elif expected < 5:
        n = max(1, round(effective))
        rng = np.random.default_rng(2025)
        extreme, remaining = 0, 9999
        batch_size = max(1, min(9999, 1_000_000 // len(levels)))
        while remaining:
            size = min(batch_size, remaining)
            simulated = rng.multinomial(n, np.full(len(levels), 1 / len(levels)), size=size)
            null = np.square(simulated - n / len(levels)).sum(axis=1) / (n / len(levels))
            extreme += np.count_nonzero(null >= statistic - 1e-12)
            remaining -= size
        pvalue = (1 + extreme) / 10000
        method = "multinomial simulation"
    else:
        pvalue = float(chisquare(observed).pvalue)
        method = "chi-square approximation"
    n = max(1, round(effective))
    band = binom.ppf([.025 / len(levels), 1 - .025 / len(levels)], n, 1 / len(levels)) / n * totals.sum()
    return {"pvalue": pvalue, "effective": effective, "method": method, "band": band,
                "balanced": pvalue >= .05, "weighted": not np.allclose(weights, weights[0])}


@recordable
@dataclass
class BalanceResult:
    export: object
    table: pd.DataFrame
    before: dict | None
    after: dict | None
    message: str
    error: bool = False


@recordable
def _analyze(source, options, card_name="targ_balance"):
    target, reason = _eligibility(source)
    if reason:
        return BalanceResult(source, pd.DataFrame(), None, None, reason)
    weight_columns = [c for c in source.columns if Role.WEIGHTING in source.role_map.roles_for(c)]
    if len(weight_columns) > 1:
        return BalanceResult(source, pd.DataFrame(), None, None,
                             "Choose at most one Weighting variable upstream.", True)
    weight = weight_columns[0] if weight_columns else None
    output_weight = "Weights"
    suffix = 2
    while output_weight in source.columns:
        output_weight = f"Weights_{suffix}"
        suffix += 1
    predictors = tuple(c for c in source.columns if source.role_map.roles_for(c) == {Role.PREDICTOR}
                       and not str(c).startswith(Card.SHADOW_PREFIX))
    mode = options.get("mode", "none")
    sampler = None
    try:
        before = _diagnostic(source.frame, target, weight)
        if mode == "none":
            successor = source
            result = source.frame
            result_weight = weight
        else:
            sampler = TargetBalanceSampler(target=target, predictors=predictors, weight=weight,
                                           output_weight=output_weight, **options)
            result, _ = sampler.fit_resample(source.frame)
            result_weight = (weight or output_weight) if mode == "reweight" else weight
            added = RoleMap.from_primitive({"weighting": [output_weight]}) if mode == "reweight" and weight is None else None
            successor = source.with_sampling_step(
                sampler, name=card_name, preview_frame=result, added_roles=added,
                operation="Balance nominal target: " + mode,
            )
        after = _diagnostic(result, target, result_weight)
        incoming = balance_weights(source.frame, weight)
        outgoing = balance_weights(result, result_weight)
        rows = []
        for c in source.frame[target].cat.categories:
            first, last = (source.frame[target] == c).to_numpy(), (result[target] == c).to_numpy()
            if not first.any():
                continue
            rows.append({"Class": str(c), "Original count": int(first.sum()),
                         "Resulting count": int(last.sum()), "Original total weight": incoming[first].sum(),
                         "Class multiplier": sampler.class_factors_[c] if mode == "reweight" else 1.,
                         "Resulting total weight": outgoing[last].sum()})
        advice = ("Already compatible with equal frequencies; prefer None unless validation supports balancing."
                  if before["balanced"] else "Unequal target frequencies; balancing remains optional and needs model validation.")
        message = f"All-data preview. {advice}"
        if mode == "resample":
            message += f" Requested {sampler.desired_count_:,} rows per class. Incoming weights are retained; equal counts need not give equal weight totals."
        if sampler and sampler.warnings_:
            message += " " + " ".join(sampler.warnings_)
        missing = int(source.frame[target].isna().sum())
        if missing:
            message += f" {missing:,} missing targets excluded from diagnostics."
        return BalanceResult(successor, pd.DataFrame(rows), before, after, message)
    except ValueError as exc:
        return BalanceResult(source, pd.DataFrame(), None, None, str(exc), True)


@recordable
def _figure(result, full_screen=False):
    if result.table.empty:
        return Card.empty_figure(result.message)
    table = result.table
    colors = [qualitative.Dark24[i % len(qualitative.Dark24)] for i in range(len(table))]
    figure = (make_subplots(rows=1, cols=3, specs=[[{"type": "domain"}, {"type": "xy"}, {"type": "domain"}]],
                           column_widths=[.22, .56, .22], subplot_titles=["Before", "Weighted totals", "After"])
              if full_screen else go.Figure())
    for label, column, opacity in [("Before", "Original total weight", .4), ("After", "Resulting total weight", 1.)]:
        trace = go.Bar(x=table["Class"], y=table[column], name=label,
                       marker_color=colors, opacity=opacity,
                       hovertemplate="%{x}: %{y:.4g}<extra>" + label + "</extra>")
        if full_screen:
            figure.add_trace(trace, row=1, col=2)
        else:
            figure.add_trace(trace)
    if full_screen:
        for column, col in [("Original total weight", 1), ("Resulting total weight", 3)]:
            figure.add_trace(go.Pie(labels=table["Class"], values=table[column], sort=False,
                                    marker_colors=colors, showlegend=False, textinfo="label+percent"), row=1, col=col)
    band = result.before["band"]
    # Shapes cannot hover. A filled trace on a hidden, normalized overlay axis
    # spans the same full plotting width without adding categorical tick values.
    figure.add_trace(go.Scatter(
        x=[0, 1, 1, 0, 0], y=[band[0], band[0], band[1], band[1], band[0]],
        xaxis="x2", yaxis="y", mode="lines", line_width=0,
        fill="toself", fillcolor="rgba(128,128,128,0.12)",
        hoveron="fills", hoverinfo="text", showlegend=False,
        text=(f"Equal-frequency band: {band[0]:.4g}–{band[1]:.4g} total weight"
              "<br>Advisory 95% simultaneous limits for equally likely classes,"
              "<br>using the original total weight."),
    ))
    figure.update_xaxes(type="category", categoryorder="array", categoryarray=table["Class"].tolist())
    figure.update_layout(xaxis2={"overlaying": "x", "anchor": "y", "type": "linear",
                                    "range": [0, 1], "visible": False, "fixedrange": True})
    figure.update_yaxes(title="Total weight", rangemode="tozero")
    figure.update_layout(
        template="plotly_white", barmode="group", showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#bbd6f8",
        margin={"l": 45, "r": 10, "t": 35 if full_screen else 10, "b": 30},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.2,
            "yanchor": "top",
            "font": {"size": 13 if full_screen else 10}
        },
        modebar={"orientation": "v"},
    )
    return figure


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = "Target balance"
    this.description = "Assess the target's imbalance and adjust it through reweignting or resampling."
    this.front = lambda: ui.TagList(
        shinywidgets.output_widget(id="Chart", fill=True, guide=this, title="Target balance", position="left",
            text="Weighted totals before and after. Color identifies the class; pale bars are before. Gray is an advisory simultaneous 95% equal-frequency band using the original total weight. Full screen adds pies."),
    )
    this.back = lambda: ui.TagList(ui.output_text("Diagnostic"), ui.output_data_frame(
        id="Summary", guide=this, title="Class accounting", position="left",
        text="Counts, original/resulting weight totals, and applied multipliers. Only Reweight changes weights. Diagnostics are advisory and do not show whether a model will improve."))
    this.footer = lambda: ui.TagList(ui.output_ui("Busy"), ui.output_text("Status"), ui.output_ui("Modes"))

    def settings():
        def select(id, label, choices, selected, text):
            return ui.input_select(id=id, label=label, choices=choices, selected=selected,
                                   guide=this, text=text, position="left")
        def numeric(id, label, value, min, max, text):
            return ui.input_numeric(id=id, label=label, value=value, min=min, max=max, step=1,
                                    guide=this, text=text, position="left")
        return ui.TagList(
            select("Up", "Upsampling", {"random": "RandomOverSampler", "smote": "SMOTENC (automatic type adaptation)"}, "random",
                   "Random duplicates complete rows and accepts missing predictors. SMOTENC uses nominal and numeric predictors; all-numeric uses SMOTE and all-categorical uses SMOTEN. Other roles come from same-class donors."),
            select("Down", "Downsampling", {"random": "RandomUnderSampler", "medoids": "K-medoids", "centroids": "ClusterCentroids", "nearmiss": "NearMiss", "stratified": "Cluster-stratified"}, "random",
                   "Medoids retain class representatives; centroids use cluster centers or nearest rows. NearMiss-1 favors points close to the smallest class. Cluster-stratified samples within roughly sqrt(n) clusters, retaining each populated cluster."),
            ui.input_slider(id="CountFraction", label="Desired count: min → max (%)", min=0, max=100, value=50, step=1,
                            guide=this, position="left", text="0 downsamples to the smallest class; 100 upsamples to the largest. Intermediate values combine both. Counts are recomputed after required-value removal in each training fold, rounded to the nearest integer."),
            select("Metric", "Numeric distance", {"euclidean": "Euclidean", "manhattan": "Manhattan"}, "euclidean",
                   "Numeric distances after optional standardization. Nominal mismatches contribute unit distance. Centroid methods use means for Euclidean and medians for Manhattan. Pure categorical SMOTEN uses its own value-difference metric."),
            ui.input_checkbox(
                id="Normalize", label="Normalize numeric distances", value=True, 
                guide=this, position="left",
                text="Learn each numeric predictor's mean and standard deviation on training rows only. Resampled values retain original units."
            ),
            select("Voting", "Centroid voting", {"hard": "Hard: original observations", "soft": "Soft: synthetic centers"}, "hard",
                   "Hard chooses the nearest row per center (donors may repeat). Soft uses numeric centers and modal nominal values; all other columns come from the nearest same-class donor."),
            numeric("Neighbors", "SMOTE / NearMiss neighbors", 5, 1, 50,
                    "Reduced automatically for small classes. SMOTE requires at least two complete rows per upsampled class."),
            numeric("MedoidLimit", "Medoid rows per class (maximum)", 2000, 10, 5000,
                    "K-medoids uses quadratic memory. Larger classes are rejected; no hidden subsampling is performed."),
            numeric("Iterations", "Medoid / centroid iterations", 30, 1, 100,
                    "Maximum alternating optimization iterations. K-medoids uses farthest-first initialization and within-cluster medoid updates, not exhaustive PAM swaps."),
            select("Evaluation", "Evaluation weights", {"none": "None", "incoming": "Incoming weights", "balanced": "Incoming × learned factors"}, "none",
                   "Saved policy for a future model evaluator: evaluation_weights(X) returns weights without resampling. No weights by default. Balanced evaluation rejects unseen or missing target classes."),
        )
    this.settings = settings

    def server(input, output, session):
        busy = this.busy()

        @this.reactable(calc=True)
        def Incoming():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(source is not None)
            return source

        @this.reactable(calc=True)
        def Eligibility():
            return _eligibility(Incoming())

        @output
        @render.ui
        def Modes():
            _, reason = Eligibility()
            with reactive.isolate():
                try:
                    selected = input.Mode() or "none"
                except SilentException:
                    selected = "none"
            return ui.tags.fieldset(ui.input_radio_buttons(
                id="Mode", label=None, choices={"none": "None", "reweight": "Reweight", "resample": "Resample"},
                selected=selected, inline=True, guide=this, position="top",
                text="Applies immediately; no commit needed. None restores the incoming data and removes this card's pipeline contribution. Results are full-data previews; saved samplers run only during model training."), disabled=bool(reason))

        @this.reactable(calc=True)
        def RawOptions():
            req(input.Neighbors() is not None, input.MedoidLimit() is not None, input.Iterations() is not None)
            return {"mode": input.Mode() or "none", "up": input.Up(), "down": input.Down(),
                        "count_fraction": float(input.CountFraction()) / 100, "metric": input.Metric(),
                        "normalize": bool(input.Normalize()), "evaluation": input.Evaluation(), "voting": input.Voting(),
                        "neighbors": int(input.Neighbors()), "medoid_limit": int(input.MedoidLimit()), "iterations": int(input.Iterations())}

        @this.reactable(calc=True)
        @this.settle(seconds=1)
        def Options():
            return RawOptions()

        @busy.track("Balancing target preview…")
        @this.extended_task
        @this.record_context
        async def Calculate(source, options):
            result = (_analyze(source, options, this.name) if Module.IS_SHINYLIVE else
                      await asyncio.to_thread(_analyze, source, options, this.name))
            return source, options, result

        @this.reactable()
        def Start():
            source, options = Incoming(), Options()
            Calculate.cancel()
            Calculate.invoke(source, options)

        @this.reactable(calc=True)
        def Results():
            try:
                source, options, result = Calculate.result()
            except SilentOperationInProgressException:
                req(False)
            current = Incoming()
            req((source is current or source.equals(current)) and options == Options() == RawOptions())
            return result

        @this.reactable(calc=True)
        def Export():
            if input.Mode() in (None, "none") or Eligibility()[1]:
                return Incoming()
            # Block downstream while a new option is settling, not just while
            # the extended task runs; never export a previous mode's preview.
            req(Options() == RawOptions())
            result = Results()
            req(not result.error)
            return result.export

        @output
        @render_widget
        @this.record_context
        def Chart():
            full = bool(this.isFullScreen())
            try:
                figure = _figure(Results(), full)
            except SilentException:
                figure = Card.empty_figure("Waiting for data or calculation.")
            widget = go.FigureWidget(figure)
            widget._config = {"displayModeBar": full, "displaylogo": False, "responsive": True}
            return widget

        @output
        @render.data_frame
        def Summary():
            table = Results().table.copy()
            for c in table.select_dtypes(include="floating"):
                table[c] = table[c].map(lambda x: f"{x:.4g}")
            return render.DataTable(table, width="100%", height=None, filters=False)

        @output
        @render.text
        def Diagnostic():
            result = Results()
            if result.before is None:
                return result.message
            pieces = []
            for label, diagnostic in [("Before", result.before), ("After", result.after)]:
                pieces.append(f"{label}: p={diagnostic['pvalue']:.4g}, effective n={diagnostic['effective']:.4g} ({diagnostic['method']}).")
            return " ".join(pieces) + " Advisory 5% test; weighted results use a Kish effective-size approximation. Resampled rows are not independent new evidence."

        @output
        @render.text
        def Status():
            return Results().message

        @output
        @render.ui
        def Busy():
            return busy.ui()

        session.on_ended(Calculate.cancel)
        return Export

    this.server = server
    return this


if Module.running_directly(name=__name__):
    frame = pd.DataFrame({"value": np.arange(100, dtype=float), "target": pd.Categorical(["a"] * 80 + ["b"] * 20)})
    this = instance()
    this._imports.set(proxy_data(_df=frame, _roles=RoleMap.from_primitive({"predictor": ["value"], "target": ["target"]})))
    this.run()
