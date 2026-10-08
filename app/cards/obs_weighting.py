"""Inspect weight components and apply explicit, training-fitted power transforms."""
from __future__ import annotations

import os
import sys
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
from plotly.subplots import make_subplots
from proxy_data import proxy_data
from roles import Role, weighting_purpose
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from weighting import WeightingError, weight_product

EFFECTIVE = "__effective__"


@recordable
def _columns(source):
    return [c for c in source.columns if Role.WEIGHTING in source.role_map.roles_for(c)]


@recordable
class WeightPowerTransformer(TransformerMixin, BaseEstimator):
    """Preserve each component's training mean; never learn from Test rows.

    Applying the same power to all components makes their product proportional
    to the original effective product raised to that power. Zeros stay zero.
    """
    def __init__(self, columns=(), selected=(), power=1., weighting_test_policy="importance"):
        self.columns = columns
        self.selected = selected
        self.power = power
        self.weighting_test_policy = weighting_test_policy

    def fit(self, X, y=None):
        if self.power not in (2., 1., .5, 1 / 3):
            raise WeightingError("Choose square, original, square root or cube root")
        if self.weighting_test_policy not in ("importance", "uniform"):
            raise WeightingError("Unknown Test weighting policy")
        if not set(self.selected).issubset(self.columns):
            raise WeightingError("The selected component is no longer assigned Weighting")
        self.scales_ = {}
        for column in self.selected:
            values = weight_product(X, (column,)).to_numpy()
            maximum = float(values.max()) if len(values) else 0.
            if not maximum or self.power == 1:
                self.scales_[column] = (1., 1.)
                continue
            scaled = values / maximum
            powered = scaled ** self.power
            self.scales_[column] = (maximum, float(scaled.mean() / powered.mean()))
        return self

    def transform(self, X):
        check_is_fitted(self, "scales_")
        result = X.copy()
        for column, (maximum, multiplier) in self.scales_.items():
            if column not in X and weighting_purpose(column) == "balance":
                continue  # Training-only balancing samplers are skipped at inference.
            values = weight_product(X, (column,)).to_numpy()
            if self.power != 1:
                try:
                    with np.errstate(over="raise", under="raise", invalid="raise"):
                        result[column] = ((values / maximum) ** self.power) * maximum * multiplier
                except FloatingPointError as error:
                    raise WeightingError("Power transform overflow or underflow; rescale the component") from error
        return result

    def evaluation_weights(self, transformed_frame):
        """Evaluate already-transformed Test rows; do not apply the power twice."""
        check_is_fitted(self, "scales_")
        return Card.observation_weights(transformed_frame, self.columns, purpose="test",
                                      test_policy=self.weighting_test_policy)


@recordable
def _apply(source, target=EFFECTIVE, power=1., test_policy="importance"):
    columns = tuple(_columns(source))
    if not columns:
        return source
    if power != 1 and target != EFFECTIVE and target not in columns:
        raise WeightingError("The selected weight component is unavailable")
    selected = columns if target == EFFECTIVE else (target,)
    if power == 1:
        selected = ()
    transformer = WeightPowerTransformer(columns, selected, power, test_policy).fit(source.frame)
    return source.with_pipeline_step(
        transformer, name="obs_weighting", operation="Configure observation weighting",
        preview_frame=transformer.transform(source.frame),
    )


@recordable
def _analyze(source, *, test=False, test_policy="importance"):
    columns = _columns(source)
    if not columns:
        return {"columns": [], "message": "No variables have the Weighting role."}
    if not len(source.frame):
        return {"columns": [], "message": "No observations to display."}
    try:
        active = columns if not test else ([] if test_policy == "uniform" else
                    [c for c in columns if weighting_purpose(c) == "importance"])
        # Validate each component, even when excluded by the Test preview policy.
        for column in columns:
            weight_product(source.frame, (column,))
        effective = weight_product(source.frame, active)
        values = effective.to_numpy()
        zeros = int(np.count_nonzero(values == 0))
        maximum = float(values.max())
        scaled = values / maximum if maximum else values
        total = scaled.sum()
        ess = float(total ** 2 / (scaled @ scaled)) if total else 0.
        share = float(scaled.max() / total) if total else 0.
        return {"columns": columns, "effective": effective, "zeros": zeros,
                "zero_fraction": zeros / len(values), "ess": ess, "largest_share": share,
                "message": "", "test": test, "policy": test_policy}
    except WeightingError as error:
        return {"columns": columns, "message": str(error)}


@recordable
def _figure(source, result, *, limit=500, bins=30, full=False):
    if result["message"]:
        return Card.empty_figure(result["message"])
    columns = result["columns"]
    effective = result["effective"].to_numpy()
    label = "Effective Test (policy preview)" if result["test"] else "Effective Training (all-data preview)"
    panels = []
    if full and len(columns) > 1:
        panels = [(c, source.frame[c].to_numpy(dtype=float)) for c in columns]
    panels.append((label, effective))
    # Normal view always shows effective bars and distribution. Full screen
    # adds component panels only when there is more than one weight.
    component_count = len(panels) - 1
    rows = 2 if component_count else 1
    grid_columns = max(2, component_count)
    height = rows * (280 if full else 180)
    titles = [name for name, _ in panels[:-1]]
    specs = [[{} for _ in range(grid_columns)] for _ in range(rows)]
    specs[-1] = [{}, {}] + [None] * (grid_columns - 2)
    titles += [label, "Effective distribution"]
    figure = make_subplots(rows=rows, cols=grid_columns, subplot_titles=titles,
                           specs=specs, horizontal_spacing=.2 / grid_columns,
                           vertical_spacing=65 / max(90, height - 90))
    # Independent row layouts: equal-width components above, 70:30 effective
    # bars/distribution below. Only the effective pair has a narrow gap.
    figure.update_xaxes(domain=[0., .7 * .975], row=rows, col=1)
    figure.update_xaxes(domain=[.7 * .975 + .025, 1.], row=rows, col=2)
    figure.layout.annotations[-2].x = .7 * .975 / 2
    figure.layout.annotations[-1].x = (.7 * .975 + .025 + 1.) / 2
    positions = np.unique(np.linspace(0, len(effective) - 1, min(int(limit), len(effective)), dtype=int))
    for index, (name, values) in enumerate(panels):
        row, col = (rows - 1, 0) if index == component_count else (0, index)
        figure.add_trace(go.Bar(x=positions + 1, y=values[positions], name=name,
                                marker_color="#4477aa", hovertemplate="Observation %{x}<br>Weight %{y:.5g}<extra>%{fullData.name}</extra>"), row=row+1, col=col+1)
        figure.update_xaxes(title_text="Observations", row=row+1, col=col+1)
        figure.update_yaxes(title_text="Weights", rangemode="tozero", row=row+1, col=col+1)
    histogram_scale = float(effective.max()) or 1.
    counts, edges = np.histogram(effective / histogram_scale, bins=int(bins), range=(0., 1.))
    edges *= histogram_scale
    row, col = rows - 1, 1
    figure.add_trace(go.Bar(x=counts, y=edges[:-1] + np.diff(edges) / 2,
                            width=np.diff(edges), orientation="h", marker_color="#ee9944",
                            customdata=np.column_stack((edges[:-1], edges[1:])),
                            hovertemplate="Weights %{customdata[0]:.5g}–%{customdata[1]:.5g}<br>Count %{x}<extra></extra>"), row=row+1, col=col+1)
    figure.update_xaxes(title_text="Counts", row=row+1, col=col+1)
    figure.update_yaxes(matches=figure.data[-2].yaxis, showticklabels=False,
                        title_text=None, rangemode="tozero", row=row+1, col=col+1)
    figure.update_layout(template="plotly_white", autosize=True,
                         showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#bbd6f8",
                         margin={"l": 50, "r": 20, "t": 45, "b": 45}, bargap=.05)
    return figure


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = "Observation Weighting"
    this.description = "Inspect weight components, effective weight concentration and zero weights; optionally reshape weights."
    this.front = lambda: ui.div(
        shinywidgets.output_widget("Chart", fill=True, guide=this, title="Weight distributions",
            position="left", text="Effective weights and their distribution are shown in normal view. Full screen adds individual component charts when there is more than one weight. The horizontal histogram counts all observations."),
        class_="obs-weighting-charts html-fill-container html-fill-item", style="min-height:0;",
    )
    this.footer = lambda: ui.TagList(
        ui.output_ui("Powers"),
        ui.output_ui("Status"),
    )
    this.settings = lambda: ui.TagList(
        ui.input_select("Component", label="Component to transform", choices={EFFECTIVE: "Effective (all components)"}, selected=EFFECTIVE,
            guide=this, position="left", text="Effective applies the same power to all components. Select one variable to change only that contribution. balance__ names identify balancing factors."),
        ui.input_select("TestPolicy", label="Test weighting", choices={"importance": "Importance only", "uniform": "Uniform (no weights)"}, selected="importance",
            guide=this, position="left", text="Saved with the pipeline. Importance excludes balance__ components. Uniform ignores all weights. This configures weight calculation; the model evaluator must request and apply it."),
        ui.input_checkbox("ShowTest", label="Preview Test weighting", value=False,
            guide=this, position="left", text="Illustrate the selected Test policy on the current observations. This is not a held-out evaluation."),
        ui.input_numeric("Limit", label="Maximum bars per variable", value=500, min=10, max=10000, step=50,
            guide=this, position="left", text="Evenly spaced observation positions limit drawing only. Histogram, zero percentage and concentration summaries use all observations."),
        ui.input_slider("Bins", label="Distribution bins", min=5, max=100, value=30, step=1,
            guide=this, position="left", text="Number of equal-width weight intervals in the distribution chart."),
    )

    def server(input, output, session):
        @this.reactable(calc=True)
        @this.record_context
        def Incoming():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(source is not None)
            return source

        @output
        @render.ui
        def Powers():
            disabled = not _columns(Incoming())
            with reactive.isolate():
                try:
                    selected = input.Power() or "1"
                except SilentException:
                    selected = "1"
            return ui.tags.fieldset(
                ui.input_radio_buttons("Power", label="Weight shape", choices={"2": "Square", "1": "Original", ".5": "Square root", str(1 / 3): "Cube root"}, selected=selected, inline=True,
                    guide=this, position="top", text="Changes downstream weights. Roots reduce relative extremes; square amplifies them. Each transformed component retains its training mean. Zeros stay zero; changing weights can alter class balance and the learning objective."),
                disabled=disabled,
            )

        @this.reactable()
        def Choices():
            choices = {EFFECTIVE: "Effective (all components)", **{c: c for c in _columns(Incoming())}}
            with reactive.isolate():
                current = input.Component()
            ui.update_select("Component", choices=choices, selected=current if current in choices else EFFECTIVE)

        @this.reactable(calc=True)
        @this.record_context
        def Applied():
            try:
                exported = _apply(Incoming(), input.Component(), float(input.Power()), input.TestPolicy())
                return exported, None
            except WeightingError as error:
                return None, str(error)

        @this.reactable(calc=True)
        @this.record_context
        def Results():
            exported, error = Applied()
            return exported, ({"message": error} if error else
                _analyze(exported, test=input.ShowTest(), test_policy=input.TestPolicy()))

        @this.reactable(calc=True)
        def Export():
            exported, error = Applied()
            if exported is None:
                raise WeightingError(error)
            return exported

        @output
        @render_widget
        @this.record_context
        def Chart():
            exported, result = Results()
            figure = _figure(exported, result, limit=max(10, min(10000, int(input.Limit() or 500))),
                             bins=input.Bins(), full=this.isFullScreen())
            widget = go.FigureWidget(figure)
            widget._config = {"displayModeBar": bool(this.isFullScreen()), "displaylogo": False}
            return widget

        @output
        @render.ui
        def Status():
            exported, result = Results()
            if result["message"]:
                return ui.span(result["message"], class_="text-warning")
            n = len(exported.frame)
            warning = " More than 10% zero weights: weighted consumers will fall back to unweighted calculations." if result["zero_fraction"] > .1 else ""
            scope = "Test policy preview" if result["test"] else "Training preview"
            scroll = " Full screen shows individual components." if len(result["columns"]) > 1 else ""
            return ui.span(f"{scope}: {result['zeros']:,}/{n:,} observations ({result['zero_fraction']:.1%}) have zero effective weight. "
                           f"Effective sample size: {result['ess']:.1f}/{n:,}; largest weight share: {result['largest_share']:.1%}."
                           + warning + scroll + " Summaries and distribution use all observations; bars may be thinned.",
                           class_="text-warning" if warning else "text-secondary")

        return Export

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    frame = pd.DataFrame({"importance": np.linspace(.1, 5., 100), "balance__weights": np.tile([1., 2.], 50)})
    source = proxy_data(_df=frame)
    for column in frame:
        source.role_map.set_roles(column, [Role.WEIGHTING])
    this._imports.set(source)
    this.run()
