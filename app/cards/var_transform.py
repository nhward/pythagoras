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

import asyncio

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from cyclic_pandas import is_cyclic
from geometry_pandas import is_geometry
from list_pandas import is_list
from module import Module
from plotly.subplots import make_subplots
from proxy_data import proxy_data
from roles import Role
from shiny import reactive, render, req, ui
from shinywidgets import render_widget
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

TRANSFORM_LABELS = {
    "Center": "Mean centre",
    "Scale": "Common spread",
    "Deskew": "Reduce skew",
}
FULL_SCREEN_HORIZONTAL_SPACING = 0.025
FULL_SCREEN_VERTICAL_SPACING = 0.06


class CommonSpreadScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Give each feature unit spread without changing its fitted mean."""

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler(with_mean=True, with_std=True).fit(X)
        self.n_features_in_ = self.scaler_.n_features_in_
        if hasattr(self.scaler_, "feature_names_in_"):
            self.feature_names_in_ = self.scaler_.feature_names_in_
        return self

    def transform(self, X):
        # StandardScaler first produces zero mean and unit spread.  Restoring
        # the fitted mean changes location back without changing that spread.
        return self.scaler_.transform(X) + self.scaler_.mean_

    def inverse_transform(self, X):
        values = np.asarray(X, dtype=float)
        standard = values - self.scaler_.mean_
        return self.scaler_.inverse_transform(standard)


class VariableTransformStep(TransformerMixin, BaseEstimator):
    """DataFrame-preserving learned transformations for selected variables."""

    def __init__(self, columns: tuple[str, ...], transforms: tuple[str, ...]):
        self.columns = columns
        self.transforms = transforms

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("VariableTransformStep requires a pandas DataFrame")
        self.pipelines_ = {}
        self.failures_ = {}
        for column in self.columns:
            if column not in X.columns:
                self.failures_[column] = "Variable is absent"
                continue
            pipeline = _build_pipeline(self.transforms)
            if pipeline is None:
                continue
            try:
                self.pipelines_[column] = pipeline.fit(X[[column]])
            except (ValueError, TypeError, FloatingPointError) as error:
                self.failures_[column] = str(error)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "pipelines_"):
            raise RuntimeError("VariableTransformStep must be fitted before use")
        result = X.copy()
        for column, pipeline in self.pipelines_.items():
            if column not in result.columns:
                raise ValueError(f"Required variable {column!r} is absent")
            transformed = pipeline.transform(result[[column]])
            result[column] = np.asarray(transformed).reshape(-1)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            self.feature_names_in_ if input_features is None else input_features,
            dtype=object,
        )


@dataclass
class DistributionAnalysis:
    frame: pd.DataFrame
    statistics: pd.DataFrame
    eligible: list[str]
    excluded: dict[str, str]
    transforms: tuple[str, ...]
    pipelines: dict[str, Pipeline]
    target: str | None
    transformer: VariableTransformStep | None

    def inverse_target(self, values) -> np.ndarray:
        """Return predictions in the target's original units."""
        if self.target is None or self.target not in self.pipelines:
            raise ValueError("No fitted target transformation is available")
        series = pd.Series(values, name=self.target)
        restored = self.pipelines[self.target].inverse_transform(series.to_frame())
        return np.asarray(restored).reshape(-1)

def _kind(series: pd.Series) -> str:
    if is_cyclic(series.dtype):
        return "cyclic"
    if is_list(series.dtype):
        return "list"
    if is_geometry(series.dtype):
        return "geometry"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    if pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype):
        return "text"
    return "unsupported"


def _role_label(data: proxy_data, column: str) -> str:
    """Return every assigned role in a stable, user-facing form."""
    roles = data.role_map.roles_for(column)
    if not roles:
        return "Unassigned"
    return ", ".join(sorted(role.value.replace("_", " ").title() for role in roles))


def _has_role_label(label: str, role: str) -> bool:
    return role in {value.strip() for value in str(label).split(",")}


def _continuous_target(data: proxy_data) -> str | None:
    """Return the sole transformable continuous target, if there is one."""
    frame = data.frame
    targets = data.role_map.columns_with_role(Role.TARGET)
    candidates: list[str] = []
    for column in frame.columns:
        if column not in targets or _kind(frame[column]) != "numeric":
            continue
        observed = pd.to_numeric(
            frame[column], errors="coerce",
        ).dropna().to_numpy(dtype=float)
        if (
            observed.size >= 3
            and np.isfinite(observed).all()
            and np.unique(observed).size >= 2
        ):
            candidates.append(str(column))
    return candidates[0] if len(candidates) == 1 else None


def _eligible_columns(
    data: proxy_data,
    *,
    include_target: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """Return transformable variables and reasons other columns are excluded."""
    frame = data.frame
    predictors = data.role_map.columns_with_role(Role.PREDICTOR)
    targets = data.role_map.columns_with_role(Role.TARGET)
    target = _continuous_target(data)
    eligible: list[str] = []
    excluded: dict[str, str] = {}

    for column in frame.columns:
        name = str(column)
        series = frame[column]
        kind = _kind(series)
        if name.startswith(Card.SHADOW_PREFIX):
            excluded[name] = "Shadow variable"
        elif column in targets and name == target and not include_target:
            excluded[name] = "Continuous target transformation is not enabled"
        elif column in targets and name == target and include_target:
            eligible.append(name)
        elif column in targets:
            excluded[name] = "A single continuous numeric target is required"
        elif column not in predictors:
            excluded[name] = "Not assigned the predictor role"
        elif kind != "numeric":
            excluded[name] = f"{kind.title()} predictors are not transformed"
        else:
            observed = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            if observed.size < 3:
                excluded[name] = "Fewer than three observed values"
            elif not np.isfinite(observed).all():
                excluded[name] = "Contains infinite values"
            elif np.unique(observed).size < 2:
                excluded[name] = "Constant predictor"
            else:
                eligible.append(name)
    return eligible, excluded


def _build_pipeline(transforms: list[str] | tuple[str, ...]) -> Pipeline | None:
    """Build the selected scikit-learn workflow in its required order."""
    selected = set(transforms)
    steps: list[tuple[str, object]] = []
    if "Deskew" in selected:
        # PowerTransformer standardises by default.  That must be disabled so
        # the card's centring and common-spread controls remain independent.
        steps.append((
            "deskew",
            PowerTransformer(method="yeo-johnson", standardize=False),
        ))
    if "Scale" in selected:
        steps.append((
            "common_spread",
            CommonSpreadScaler(),
        ))
    if "Center" in selected:
        steps.append((
            "centre",
            StandardScaler(
                with_mean=True,
                with_std=False,
            ),
        ))
    if not steps:
        return None
    return Pipeline(steps).set_output(transform="pandas")


def _describe(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return {
        "Mean": float(numeric.mean()),
        "Standard deviation": float(numeric.std(ddof=0)),
        "Skew": float(numeric.skew()),
        # pandas reports Fisher (excess) kurtosis.  Adding three gives the
        # ordinary Pearson definition, for which a normal distribution is 3.
        "Kurtosis": float(numeric.kurt() + 3.0),
    }


def _transform_name(transforms: list[str] | tuple[str, ...]) -> str:
    selected = set(transforms)
    return " → ".join(
        TRANSFORM_LABELS[value]
        for value in ("Deskew", "Scale", "Center")
        if value in selected
    ) or "None"


def _analyse_distribution(
    data: proxy_data,
    transforms: list[str] | tuple[str, ...],
    include_target: bool = False,
) -> DistributionAnalysis:
    """Transform eligible columns from a fresh source frame and summarise them."""
    source = data.frame
    eligible, excluded = _eligible_columns(data, include_target=include_target)
    target = _continuous_target(data) if include_target else None
    selected = list(transforms)
    transformer = (
        VariableTransformStep(tuple(eligible), tuple(selected)).fit(source)
        if selected else None
    )
    frame = transformer.transform(source) if transformer is not None else source.copy()
    pipelines = {} if transformer is None else transformer.pipelines_
    failures = {} if transformer is None else transformer.failures_
    rows: list[dict[str, object]] = []

    for column in eligible:
        before = source[column]
        after = frame[column]
        fitted_lambda = np.nan
        status = _transform_name(selected)
        pipeline = pipelines.get(column)
        if column in failures:
            status = f"Not transformed: {failures[column]}"
            excluded[column] = status
        elif pipeline is not None and "deskew" in pipeline.named_steps:
            fitted_lambda = float(pipeline.named_steps["deskew"].lambdas_[0])

        before_stats = _describe(before)
        after_stats = _describe(after)
        rows.append({
            "Variable": str(column),
            "Role": _role_label(data, column),
            "Mean before": before_stats["Mean"],
            "Mean after": after_stats["Mean"],
            "SD before": before_stats["Standard deviation"],
            "SD after": after_stats["Standard deviation"],
            "Skew before": before_stats["Skew"],
            "Skew after": after_stats["Skew"],
            "Kurtosis before": before_stats["Kurtosis"],
            "Kurtosis after": after_stats["Kurtosis"],
            "Skew reduction": abs(before_stats["Skew"]) - abs(after_stats["Skew"]),
            "Yeo-Johnson lambda": fitted_lambda,
            "Transformation": status,
        })

    for column, reason in excluded.items():
        if column in eligible:
            continue
        rows.append({
            "Variable": str(column),
            "Role": _role_label(data, column),
            "Mean before": np.nan,
            "Mean after": np.nan,
            "SD before": np.nan,
            "SD after": np.nan,
            "Skew before": np.nan,
            "Skew after": np.nan,
            "Kurtosis before": np.nan,
            "Kurtosis after": np.nan,
            "Skew reduction": np.nan,
            "Yeo-Johnson lambda": np.nan,
            "Transformation": reason,
        })

    statistics = pd.DataFrame(rows)
    if not statistics.empty:
        statistics = statistics.sort_values(
            ["Skew reduction", "Variable"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
    return DistributionAnalysis(
        frame, statistics, eligible, excluded, tuple(selected), pipelines, target,
        transformer,
    )


def _apply_analysis(
    source: proxy_data,
    analysis: DistributionAnalysis,
    *,
    step_name: str = "var_transform",
    operation: str = "Variable Transform",
) -> proxy_data:
    """Register the selected transformations and their display preview."""
    if not analysis.transforms:
        return source.with_inactive_step(
            stage="Learning",
            card=step_name,
            operation=operation,
        )
    return source.with_pipeline_step(
        analysis.transformer,
        name=step_name,
        operation=operation,
        preview_frame=analysis.frame,
    )


def _axis_range(
    values: pd.Series,
    *,
    anchors: tuple[float, ...] = (),
) -> list[float]:
    """Return a padded range determined only by original statistics."""
    finite = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    combined = np.r_[finite, np.asarray(anchors, dtype=float)]
    combined = combined[np.isfinite(combined)]
    if not combined.size:
        return [0.0, 1.0]
    lower = float(combined.min())
    upper = float(combined.max())
    if lower == upper:
        width = max(abs(lower) * 0.1, 1.0)
    else:
        width = (upper - lower) * 0.08
    lower -= width
    upper += width
    return [lower, upper]


def _add_distribution_panel(
    figure: go.Figure,
    eligible: pd.DataFrame,
    *,
    before_x: str,
    before_y: str,
    after_x: str,
    after_y: str,
    row: int,
    col: int,
    transformed: bool,
    labels: bool,
    show_legend: bool,
) -> None:
    if transformed:
        for _, item in eligible.iterrows():
            is_target = _has_role_label(item["Role"], "Target")
            figure.add_trace(go.Scatter(
                x=[item[before_x], item[after_x]],
                y=[item[before_y], item[after_y]],
                mode="lines",
                line={
                    "color": "rgba(180,95,6,0.55)" if is_target
                    else "rgba(80,90,105,0.35)",
                    "width": 2.5 if is_target else 1.5,
                },
                hoverinfo="skip",
                showlegend=False,
            ), row=row, col=col)
    figure.add_trace(go.Scatter(
        x=eligible[before_x],
        y=eligible[before_y],
        mode="markers+text" if labels else "markers",
        text=eligible["Variable"] if labels else None,
        textposition="top center",
        name="Before",
        legendgroup="before",
        showlegend=show_legend,
        marker={
            "color": "#090a0a",
            "size": [
                12 if _has_role_label(role, "Target") else 9
                for role in eligible["Role"]
            ],
            "symbol": [
                "diamond-open" if _has_role_label(role, "Target") else "circle-open"
                for role in eligible["Role"]
            ],
        },
        customdata=eligible[["Variable", "Role", "Transformation"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>x: %{x:.4g}<br>y: %{y:.4g}"
            "<br>Role: %{customdata[1]}<br>%{customdata[2]}"
            "<extra>Before</extra>"
        ),
    ), row=row, col=col)
    if transformed:
        figure.add_trace(go.Scatter(
            x=eligible[after_x],
            y=eligible[after_y],
            mode="markers+text" if labels else "markers",
            text=eligible["Variable"] if labels else None,
            textposition="bottom center",
            name="After",
            legendgroup="after",
            showlegend=show_legend,
            marker={
                "color": [
                    "#b45f06" if _has_role_label(role, "Target") else "#154c79"
                    for role in eligible["Role"]
                ],
                "size": [
                    12 if _has_role_label(role, "Target") else 10
                    for role in eligible["Role"]
                ],
                "symbol": [
                    "diamond" if _has_role_label(role, "Target") else "circle"
                    for role in eligible["Role"]
                ],
            },
            customdata=eligible[["Variable", "Role", "Transformation"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>x: %{x:.4g}<br>y: %{y:.4g}"
                "<br>Role: %{customdata[1]}<br>%{customdata[2]}"
                "<extra>After</extra>"
            ),
        ), row=row, col=col)
    if show_legend and eligible["Role"].map(
        lambda role: _has_role_label(role, "Target")
    ).any():
        figure.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="Target variable",
            marker={"color": "#b45f06", "size": 11, "symbol": "diamond"},
            hoverinfo="skip", showlegend=True,
        ), row=row, col=col)


def _distribution_figure(
    statistics: pd.DataFrame,
    *,
    transformed: bool,
    labels: bool = False,
    full_screen: bool = False,
) -> go.Figure:
    eligible = statistics.loc[statistics["Mean before"].notna()].copy()
    if eligible.empty:
        return Card.empty_figure("There are no continuous numeric predictors to chart")

    mean_range = _axis_range(eligible["Mean before"], anchors=(0.0,))
    # Retain zero as the meaningful baseline while leaving enough display
    # space below it for markers centred on the axis to remain unclipped.
    spread_range = _axis_range(eligible["SD before"], anchors=(0.0, 1.0))
    skew_range = _axis_range(eligible["Skew before"], anchors=(0.0,))
    kurtosis_range = _axis_range(eligible["Kurtosis before"], anchors=(3.0,))

    if full_screen:
        figure = make_subplots(
            rows=2,
            cols=2,
            specs=[[{}, {}], [{}, None]],
            subplot_titles=(
                "Skew and spread", "Location and spread", "Distribution shape",
            ),
            shared_xaxes="columns",
            shared_yaxes="rows",
            horizontal_spacing=FULL_SCREEN_HORIZONTAL_SPACING,
            vertical_spacing=FULL_SCREEN_VERTICAL_SPACING,
        )
        panels = [
            ("Skew before", "SD before", "Skew after", "SD after", 1, 1),
            ("Mean before", "SD before", "Mean after", "SD after", 1, 2),
            (
                "Skew before", "Kurtosis before",
                "Skew after", "Kurtosis after", 2, 1,
            ),
        ]
        for index, (before_x, before_y, after_x, after_y, row, col) in enumerate(panels):
            _add_distribution_panel(
                figure, eligible,
                before_x=before_x, before_y=before_y,
                after_x=after_x, after_y=after_y,
                row=row, col=col, transformed=transformed, labels=labels,
                show_legend=index == 0,
            )
        figure.update_xaxes(title_text="", range=skew_range, row=1, col=1)
        figure.update_yaxes(
            title_text="Standard deviation", range=spread_range, row=1, col=1,
        )
        figure.update_xaxes(title_text="Mean", range=mean_range, row=1, col=2)
        figure.update_yaxes(range=spread_range, row=1, col=2)
        figure.update_xaxes(title_text="Skew", range=skew_range, row=2, col=1)
        figure.update_yaxes(title_text="Kurtosis", range=kurtosis_range, row=2, col=1)
    else:
        figure = make_subplots(
            rows=1, cols=1, subplot_titles=("Location and spread",),
        )
        _add_distribution_panel(
            figure, eligible,
            before_x="Mean before", before_y="SD before",
            after_x="Mean after", after_y="SD after",
            row=1, col=1, transformed=transformed, labels=labels,
            show_legend=True,
        )
        figure.update_xaxes(title_text="Mean", range=mean_range, row=1, col=1)
        figure.update_yaxes(
            title_text="Standard deviation", range=spread_range, row=1, col=1,
        )

    figure.update_xaxes(fixedrange=not full_screen)
    figure.update_yaxes(fixedrange=not full_screen)
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#bbd6f8",
        margin={"l": 65, "r": 25, "t": 45, "b": 75},
        legend={
            "orientation": "h", "x": 0.5, "xanchor": "center",
            "y": -0.22, "yanchor": "top",
        },
        modebar={"orientation": "v"},
    )
    return figure


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = "Variable Transform"
    this.description = "This card explores and applies transformations to the location, spread, and shape of continuous numeric predictors and, when explicitly enabled, one continuous target."

    def front():
        return ui.TagList(
            ui.span(
                "Variables in location–spread and distribution-shape space",
                class_="text-primary text-center d-block",
            ),
            shinywidgets.output_widget(
                id="DistributionChart", fill=True, guide=this,
                title="Distribution transformation map",
                text="The card maps mean against standard deviation. Full-screen mode adds skew against standard deviation and kurtosis against skew. When transformations are selected, lines connect each original point to its transformed position without changing the original axis scales.",
                position="left",
            ),
        )
    this.front = front

    def back():
        return ui.TagList(
            ui.span(
                "Distribution statistics before and after transformation",
                class_="text-primary text-center d-block",
            ),
            ui.output_ui(
                id="Statistics", guide=this, title="Distribution statistics",
                text="The table reports the moments before and after transformation, the fitted Yeo–Johnson lambda, and explicit reasons for excluded variables.",
                position="left",
            ),
        )
    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(id="Busy"),
            ui.output_ui(id="Check"),
            ui.input_checkbox_group(
                id="Transform", label="Transform variables",
                choices=TRANSFORM_LABELS, selected=[], inline=True,
                guide=this, title="Apply transformations", position="top",
                text="Reduce skew with a fitted Yeo–Johnson power transform, give predictors a common standard deviation, or mean-centre them. Selected operations are always applied in that order, and clearing the choices restores the incoming data.",
            ),
            class_="vertically-scrollable-footer",
        )
    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_checkbox(
                id="IncludeTarget",
                label="Include a continuous numeric target",
                value=False,
                guide=this,
                title="Transform the continuous target",
                position="left",
                text="When exactly one continuous numeric target is assigned, include it in the selected transformations. Otherwise this setting has no effect. Target transformations will eventually become part of the persisted proxy workflow; until then, interpret transformed target values with care.",
            ),
            ui.input_checkbox(
                id="Labels", label="Label chart points", value=False,
                guide=this, title="Point labels", position="left",
                text="Show variable names beside chart points. Hover information remains available when labels are hidden.",
            ),
            ui.input_slider(
                id="Digits", label="Table decimal places", min=1, max=8,
                value=3, step=1, guide=this, title="Decimal places",
                position="left", text="Controls numeric rounding on the flip-side table.",
            ),
        )
    this.settings = settings

    def server(input, output, session):
        busy = this.busy()

        @this.suspendable(calc=True)
        def incomingproxy_data():
            req(this._imports.is_set())
            return this._imports.get()

        @this.settle(seconds=1)
        @this.suspendable(calc=True)
        def Options():
            selected = list(input.Transform() or [])
            target = _continuous_target(incomingproxy_data())
            return {
                "transforms": [
                    name for name in ("Deskew", "Scale", "Center")
                    if name in selected
                ],
                "include_target": target is not None and bool(input.IncludeTarget()),
            }

        @busy.track("Transforming continuous numeric predictors…")
        @reactive.extended_task
        async def Calculate(data: proxy_data, options: dict[str, object]):
            return await asyncio.to_thread(_analyse_distribution, data, **options)

        @this.suspendable()
        def StartAnalysis():
            Calculate.invoke(incomingproxy_data().clone(), Options())

        @this.suspendable(calc=True)
        @this.record_code
        def Analysis():
            return Calculate.result()

        @this.suspendable(calc=True)
        @this.record_code
        def TransformedData():
            source = incomingproxy_data()
            analysis = Analysis()
            return _apply_analysis(
                source,
                analysis,
                step_name=this.namespace,
                operation=this.long_name,
            )

        @this.suspendable(triggers=[TransformedData])
        def export():
            this._exports.set(TransformedData())

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render_widget
        def DistributionChart():
            full_screen = bool(this.isFullScreen())
            analysis = Analysis()
            figure = _distribution_figure(
                analysis.statistics,
                transformed=bool(analysis.transforms),
                labels=bool(input.Labels()),
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
        def Statistics():
            return ui.output_data_frame(id="StatisticsTable")

        @output
        @render.data_frame
        def StatisticsTable():
            digits = int(input.Digits())
            table = Analysis().statistics.copy()
            numeric = table.select_dtypes(include="number").columns
            table[numeric] = table[numeric].round(digits)
            return render.DataTable(table, width="100%", height="98%")

        @output
        @render.ui
        def Check():
            analysis = Analysis()
            target_count = int(analysis.target is not None)
            predictor_count = len(analysis.eligible) - target_count
            subject = f"{predictor_count} predictors"
            if target_count:
                subject += " and 1 target"
            # excluded = len(analysis.excluded)
            # suffix = f"; {excluded} excluded variables are explained on the flip-side" if excluded else ""
            if analysis.transforms:
                return ui.span(
                    f"Transformed {subject} using {_transform_name(analysis.transforms)}.",
                    class_="text-success",
                )
            return ui.span(
                f"{subject} are continuous numeric variables.",
                class_="text-primary",
            )

        session.on_ended(Calculate.cancel)

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
