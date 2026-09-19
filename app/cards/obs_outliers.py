from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import asyncio
from dataclasses import dataclass, field
from threading import Event

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from module import Module
from proxy_data import proxy_data
from roles import Role
from shiny import render, ui
from shiny.types import SilentException
from shinywidgets import render_widget
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

METHODS = ("Mahalanobis", "Local Outlier Factor", "One-class SVM", "Isolation Forest", "Cook's distance")
COLORS = ("#4477aa", "#a3753d", "#ded939", "#cc66c9", "#677022")


@dataclass
class OutlierAnalysis:
    raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    ranks: pd.DataFrame = field(default_factory=pd.DataFrame)
    observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictors: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    total: int = 0
    eligible: int = 0
    message: str = ""


def _numeric(series):
    return pd.api.types.is_numeric_dtype(series.dtype) and not (
        pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_complex_dtype(series.dtype)
    )


def _predictors(data):
    return [c for c in data.columns if Role.PREDICTOR in data.role_map.roles_for(c)
            and not data.role_map.roles_for(c) & {Role.TARGET, Role.IDENTIFIER, Role.WEIGHTING}
            and not str(c).startswith(Card.SHADOW_PREFIX) and _numeric(data.frame[c])]


def _percentiles(values):
    """Equal scores stay tied; constant methods supply no ranking evidence."""
    values = pd.Series(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values.iloc[0], rtol=1e-10, atol=1e-12):
        return np.zeros(len(values))
    return ((values.rank(method="average") - 1) * 100 / (len(values) - 1)).to_numpy()


def _cooks(x, target):
    import statsmodels.api as sm
    design = sm.add_constant(x, has_constant="add")
    if np.linalg.matrix_rank(design) != design.shape[1] or len(x) <= design.shape[1] + 1:
        raise ValueError("needs a full-rank regression with residual degrees of freedom")
    model = sm.OLS(target, design).fit()
    if model.mse_resid <= np.finfo(float).eps * max(1., float(np.var(target))):
        raise ValueError("regression is a perfect or near-perfect fit")
    return model.get_influence().cooks_distance[0]


def _analyze(data, *, neighbors=20, nu=.1, trees=100, limit=3000, methods=METHODS, cancel=None):
    result = OutlierAnalysis()
    if data is None:
        result.message = "No incoming data."
        return result
    result.total = len(data.frame)
    columns = _predictors(data)
    if not columns:
        result.message = "No numeric Predictor-role variables. Encode other predictors upstream."
        return result
    if len(columns) > 200:
        result.message = "More than 200 numeric predictors; select or reduce dimensions upstream."
        return result
    frame = data.frame[columns].replace([np.inf, -np.inf], np.nan)
    usable = [c for c in columns if frame[c].nunique(dropna=True) > 1]
    if len(usable) != len(columns):
        result.notes.append(f"{len(columns) - len(usable)} constant or entirely missing predictors ignored.")
    if not usable:
        result.message = "No varying numeric predictors."
        return result
    result.predictors = usable
    matrix = frame[usable].to_numpy(dtype=float, na_value=np.nan)
    positions = np.flatnonzero(np.isfinite(matrix).all(axis=1))
    result.eligible = len(positions)
    if len(positions) < 5:
        result.message = "At least five complete, finite observations are needed."
        return result
    if len(positions) > max(5, int(limit)):
        positions = np.sort(np.random.default_rng(2025).choice(positions, max(5, int(limit)), replace=False))
        result.notes.append("Reproducible sample analyzed; unsampled observations have not been assessed.")
    x = StandardScaler().fit_transform(matrix[positions])
    if not np.isfinite(x).all():
        result.message = "Predictor magnitudes cannot be safely standardized; rescale upstream."
        return result
    result.observations = pd.DataFrame({"Row": positions + 1})
    identifiers = [c for c in data.columns if Role.IDENTIFIER in data.role_map.roles_for(c)]
    if identifiers:
        result.observations["Identifier"] = data.frame[identifiers[0]].iloc[positions].astype("string").fillna("(missing)").to_numpy()
        result.notes.append(f"Identifier: {identifiers[0]} (row numbers distinguish repeated identifiers).")
    if result.total != result.eligible:
        result.notes.append(f"{result.total - result.eligible} rows with missing/nonfinite predictors excluded from analysis only.")
    if data.role_map.columns_with_role(Role.WEIGHTING):
        result.notes.append("Equal observation importance; assigned weights are not used by these diagnostics.")
    raw = {}
    for method in METHODS:
        if cancel is not None and cancel.is_set():
            return OutlierAnalysis(message="Calculation superseded.")
        if method not in methods:
            continue
        try:
            if method == "Mahalanobis":
                model = LedoitWolf().fit(x)
                values = np.sqrt(np.maximum(0, model.mahalanobis(x)))
            elif method == "Local Outlier Factor":
                model = LocalOutlierFactor(n_neighbors=min(max(2, int(neighbors)), len(x)-1), n_jobs=1).fit(x)
                values = -model.negative_outlier_factor_
            elif method == "One-class SVM":
                model = OneClassSVM(nu=float(nu), gamma="scale", cache_size=128).fit(x)
                values = -model.score_samples(x)
            elif method == "Isolation Forest":
                model = IsolationForest(n_estimators=int(trees), max_samples=min(256, len(x)), random_state=2025, n_jobs=1).fit(x)
                values = -model.score_samples(x)
            else:
                targets = [c for c in data.columns if Role.TARGET in data.role_map.roles_for(c)]
                if len(targets) != 1 or not _numeric(data.frame[targets[0]]):
                    raise ValueError("requires one numeric Target (OLS influence only)")
                y = data.frame[targets[0]].iloc[positions].to_numpy(dtype=float, na_value=np.nan)
                if not np.isfinite(y).all():
                    raise ValueError("Target has missing/nonfinite values in analyzed rows")
                values = _cooks(x, y)
            if not np.isfinite(values).all():
                raise ValueError("returned nonfinite scores")
            raw[method] = values
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            result.notes.append(f"{method} unavailable: {error}.")
    result.raw = pd.DataFrame(raw)
    result.ranks = pd.DataFrame({method: _percentiles(values) for method, values in raw.items()})
    if not raw:
        result.message = "No usable evaluation methods. Select methods or check their requirements."
    return result


def _table(result, *, raw=False):
    if result.ranks.empty:
        return pd.DataFrame(columns=["Row", "Aggregate", "Disagreement"])
    table = result.observations.copy()
    table["Aggregate"] = result.ranks.mean(axis=1)
    table["Disagreement"] = result.ranks.max(axis=1) - result.ranks.min(axis=1)
    return pd.concat([table, result.raw if raw else result.ranks], axis=1)


def _figure(result, *, view="Aggregate", top=30, full_screen=False):
    if result.message or result.ranks.empty:
        return Card.empty_figure(result.message or "No scores available.")
    if view != "Aggregate" and view not in result.raw:
        return Card.empty_figure(f"{view} unavailable; see the status below.")
    table = _table(result)
    scores = table["Aggregate"] if view == "Aggregate" else result.raw[view]
    selected = scores.sort_values(ascending=False, kind="stable").head(int(top)).index
    rows = table.loc[selected, "Row"].astype(str).tolist()
    labels = table.loc[selected, "Identifier"].astype(str).tolist() if "Identifier" in table else [f"Row {r}" for r in rows]
    figure = go.Figure()
    for i, method in enumerate(METHODS):
        if method not in result.raw or (view != "Aggregate" and view != method):
            continue
        y = result.ranks.loc[selected, method] / len(result.ranks.columns) if view == "Aggregate" else result.raw.loc[selected, method]
        custom = np.column_stack([labels, rows, result.raw.loc[selected, method], result.ranks.loc[selected, method]])
        figure.add_bar(x=rows, y=y, name=method, marker_color=COLORS[i], customdata=custom,
                       hovertemplate="%{customdata[0]} (row %{customdata[1]})<br>Raw: %{customdata[2]:.4g}<br>Percentile: %{customdata[3]:.2f}<extra>%{fullData.name}</extra>")
    figure.update_layout(
        template="plotly_white", 
        barmode="stack", 
        showlegend=full_screen,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.0, "yanchor": "bottom"},
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="#bbd6f8",
        margin={"l": 15, "r": 25, "t": 0, "b": 35},
        xaxis={"type": "category", "categoryorder": "array", "categoryarray": rows, "tickmode": "array", "tickvals": rows, "ticktext": labels, "title": "Observation", "fixedrange": not full_screen},
        yaxis={"title": "Percentile" if view == "Aggregate" else "Raw outlier score", "fixedrange": not full_screen},
        modebar={"orientation": "v"})
    if view == "Aggregate":
        figure.update_yaxes(range=[0, 100])
    return figure


def instance():
    this = Card(file=__file__, mutable=False)
    this.long_name = "Observation outliers"
    this.description = "Compare complementary outlier scores to investigate unusual observations and disagreements between methods."

    def front():
        panels = []
        for i, name in enumerate(("Aggregate",) + METHODS):
            panels.append(ui.nav_panel(name, ui.span(name + " outlier scores", class_="text-primary text-center d-block"),
                shinywidgets.output_widget(id=f"Chart{i}", fill=True, guide=this, title=name,
                    text="Bars identify unusual observations, not errors. Aggregate stacks equal-weight percentile contributions; hover shows each raw score and percentile. Each tab shows its own highest-scoring observations in descending order; Aggregate sorts by the combined score.", position="left"), value=name))
        return ui.navset_bar(*panels, id="View", selected="Aggregate", title=None, padding=0)
    this.front = front

    this.back = lambda: ui.TagList(
        ui.span("Observation scores", class_="text-primary text-center d-block"),
        ui.input_checkbox(
            id="Raw", label="Show raw method scores", value=False, 
            guide=this, position = "left",
            text="Switch between raw scores and percentiles. Aggregate is always the mean percentile; Disagreement is the largest minus smallest percentile. Neither is an outlier probability."
        ),
        ui.output_data_frame(
            id="Scores", 
            guide=this, title="Investigation table", position = "left",
            text="All analyzed observations, sorted by the active chart's score in descending order (Aggregate when that method is unavailable). Row is the one-based position in the incoming data; Identifier is shown when assigned. No rows are removed."))
    this.footer = lambda: ui.TagList(
        ui.output_ui("Busy"), 
        ui.output_text("Status")
    )

    def settings():
        def slider(id, label, low, high, value, step, text, ticks=True, pre=None):
            return ui.input_slider(id, label=label, min=low, max=high, value=value, step=step, guide=this, text=text, position="left", ticks=ticks, pre=pre)

        return ui.TagList(
            ui.input_checkbox_group(
                id="Methods", label="Evaluation methods", choices=list(METHODS), selected=list(METHODS), 
                guide=this, position = "left",
                text="Enabled, usable methods contribute equally after percentile ranking. Mahalanobis uses shrinkage covariance. Cook's distance requires a complete numeric Target and a full-rank linear regression. Weights are not used. Select several methods to compare disagreement."
            ),
            slider("Top", "Observations displayed", 5, 100, 30, 1, "Show the highest-scoring observations for each panel, in descending order. This changes only the plot, not fitting or the full table."),
            slider("Neighbors", "LOF nearest neighbors", 2, 100, 20, 1, "Neighborhood size, capped below the analyzed row count. Small neighborhoods emphasize local anomalies; large ones compare a wider context."),
            slider("Nu", "One-class SVM nu", .02, .5, .1, 0.02, "Controls the SVM boundary: an upper bound on training errors and lower bound on support vectors, not an estimated probability of bad data."),
            slider("Trees", "Isolation Forest trees", 25, 300, 100, 1, "More trees stabilize rankings but take longer. Each tree uses up to 256 sampled rows with a fixed random seed."),
            slider("Limit", "Maximum observations to assess", 2, 7, 3, 1, "Complete rows above this cap are sampled reproducibly. Unselected rows are not assessed and rare outliers may be missed. Raising the cap can substantially increase SVM and neighborhood computation time.", True, "10^"),
        )
    this.settings = settings

    def server(input, output, session):
        busy = this.busy()
        cancellation = Event()
        analyze = this.record_code(_analyze)

        @this.reactable(calc=True)
        @this.settle(seconds=2)
        def Options():
            return {
                "neighbors": int(input.Neighbors()), 
                "nu": float(input.Nu()), 
                "trees": int(input.Trees()),
                "limit": int(10**input.Limit()), 
                "methods": tuple(input.Methods() or ())
            }

        @busy.track("Comparing observation outlier scores…")
        @this.extended_task
        async def Calculate(data, options, cancelled):
            try:
                result = await asyncio.to_thread(analyze, data, **options, cancel=cancelled)
                return data, options, result
            except Exception as error:
                this.log.exception("Outlier analysis failed")
                return data, options, OutlierAnalysis(message=f"Unable to calculate outlier scores: {error}")

        @this.reactable()
        def Start():
            nonlocal cancellation
            try:
                data = this.input_data()
            except SilentException:
                data = None
            options = Options()
            cancellation.set()
            Calculate.cancel()
            cancellation = Event()
            Calculate.invoke(data.clone() if data is not None else None, options, cancellation)

        @this.reactable(calc=True)
        def Results():
            try:
                current = this.input_data()
            except SilentException:
                current = None
            if current is None:
                return OutlierAnalysis(message="No incoming data.")
            try:
                data, options, result = Calculate.result()
                if data is None or not data.equals(current) or options != Options():
                    return OutlierAnalysis(message="Calculating outlier scores…")
                return result
            except SilentException:
                return OutlierAnalysis(message="Calculating outlier scores…")

        for i, name in enumerate(("Aggregate",) + METHODS):
            def register(i, name):
                @output(id=f"Chart{i}")
                @render_widget
                def chart():
                    full = bool(this.isFullScreen())
                    widget = go.FigureWidget(_figure(Results(), view=name, top=input.Top(), full_screen=full))
                    widget._config = {"displayModeBar": full, "displaylogo": False, "responsive": True}
                    return widget
            register(i, name)

        @output
        @render.data_frame
        def Scores():
            result = Results()
            table = _table(result, raw=bool(input.Raw()))
            if not table.empty:
                view = input.View()
                scores = result.raw[view] if view in result.raw else table["Aggregate"]
                table = table.loc[scores.sort_values(ascending=False, kind="stable").index]
            return render.DataTable(table.round(4), width="100%", height=None, filters=True)

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render.text
        def Status():
            r = Results()
            if r.message:
                return " ".join([r.message] + r.notes)
            return (f"{len(r.raw):,} of {r.total:,} observations analyzed ({r.eligible:,} complete); "
                    f"{len(r.predictors)} numeric predictors; {len(r.raw.columns)} methods. "
                    "Investigate unusual values; no observations removed. " + " ".join(r.notes))

        def stop():
            cancellation.set()
            Calculate.cancel()

        session.on_ended(stop)

        return this.input_data

    this.server = server

    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
