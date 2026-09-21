"""Role-aware diagnostics for serial dependence and longitudinal structure."""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
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
from proxy_data import proxy_data
from roles import Role
from scipy.stats import chi2
from shiny import render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget
from statsmodels.stats.multitest import multipletests

COLUMNS = ["Entity", "Variable", "Rows", "Observed", "Imputed", "Leading trimmed", "Analyzed rows", "Lags", "Lag 1", "Ljung–Box p", "Squared p", "Adjusted p", "Status"]
MAX_ROWS = 100_000
MAX_SERIES = 2_000
MAX_LAG_PRODUCTS = 20_000_000


FINDING_COLOURS = {
    "Flagged": "#d95f02",
    "Flagged (imputed; exploratory)": "#e6ab02",
    "Not flagged": "#3978a8",
    "Unavailable": "#9aa7b2",
}
FINDING_CLASSES = dict(zip(FINDING_COLOURS, [
    "obs-dependence-flagged-row", "obs-dependence-exploratory-row",
    "obs-dependence-not-flagged-row", "obs-dependence-unavailable-row",
]))


@recordable
def _display_table(table, alpha):
    table = table.copy().reset_index(drop=True)
    table["Finding"] = [
        "Unavailable" if pd.isna(p) else
        "Flagged (imputed; exploratory)" if p < alpha and imputed else
        "Flagged" if p < alpha else "Not flagged"
        for p, imputed in zip(table["Adjusted p"], table["Imputed"])
    ]
    return table


def _row_styles(table):
    return [{"rows": np.flatnonzero(table["Finding"].eq(finding)).tolist(), "class": class_name}
            for finding, class_name in FINDING_CLASSES.items()
            if table["Finding"].eq(finding).any()]


@recordable
@dataclass
class Result:
    table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COLUMNS))
    notes: list[str] = field(default_factory=list)


@recordable
def _numeric(series):
    return pd.api.types.is_numeric_dtype(series.dtype) and not (
        pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_complex_dtype(series.dtype)
    )


@recordable
def _ljung_box(values, lags):
    """Direct bounded-lag ACF avoids computing an entire quadratic correlation."""
    # Scaling first prevents overflow on large finite values.
    x = values / (float(np.max(np.abs(values))) or 1.)
    x = x - x.mean()
    denominator = np.dot(x, x)
    if denominator <= 0:
        return np.nan, np.nan
    n = len(x)
    acf = np.array([np.dot(x[k:], x[:-k]) / denominator for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    return float(chi2.sf(q, lags)), float(acf[0])


@recordable
def _rolling_impute(values, window):
    """Fill from original finite observations in the preceding window positions.

    Never reuse filled values or look ahead. Trim only the missing prefix;
    unresolved internal/trailing gaps remain missing rather than closing time.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    start = int(np.flatnonzero(finite)[0]) if finite.any() else len(values)
    original = values[start:].copy()
    original[~np.isfinite(original)] = np.nan
    if not len(original):
        return original, 0, start
    scale = float(np.nanmax(np.abs(original))) or 1.
    series = pd.Series(original / scale)
    means = series.shift(1).rolling(max(1, int(window)), min_periods=1).mean()
    filled = series.fillna(means).to_numpy() * scale
    count = int((~np.isfinite(original) & np.isfinite(filled)).sum())
    return filled, count, start


@recordable
def _analyze(source, lags=10, row_order=False, use_entities=True, use_target=True,
             imputation="rolling", window=5):
    result = Result()
    frame = source.frame
    roles = source.role_map
    keys = [c for c in frame if roles.has_role(c, Role.IDENTIFIER)] if use_entities else []
    sequences = [c for c in frame if roles.has_role(c, Role.SEQUENCE)]
    sequence = sequences[0] if len(sequences) == 1 and not row_order else None
    result.notes.append("Observation weights are not used; data and roles pass through unchanged.")
    if frame.empty:
        result.notes.append("No observations to assess.")
        return result
    if not row_order and len(sequences) != 1:
        result.notes.append("Assign exactly one Sequence role in Variable roles, or explicitly select exploratory row order. A time-series design cannot be inferred from values alone.")
        return result
    if row_order:
        result.notes.append("Exploratory row order: results concern current row positions, not established time order. Assign a Sequence role for time-based analysis.")
    elif not (_numeric(frame[sequence]) or pd.api.types.is_datetime64_any_dtype(frame[sequence])):
        result.notes.append("Sequence must be numeric or datetime. Convert date strings upstream; categorical or textual ordering is ambiguous.")
        return result
    required = keys + ([sequence] if sequence else [])
    if required and frame[required].isna().any(axis=None):
        result.notes.append("Missing entity or sequence keys: resolve these before analysis; dropping them could bridge unknown intervals.")
        return result
    if sequence and _numeric(frame[sequence]) and not np.isfinite(frame[sequence].to_numpy(dtype=float)).all():
        result.notes.append("Sequence contains nonfinite values; resolve these before analysis.")
        return result
    if sequence and frame.duplicated(subset=required).any():
        result.notes.append("Entity + sequence keys are not unique (or sequence alone without entities). Resolve tied observations or assign the entity Identifier role before testing.")
        return result
    if keys:
        groups = list(frame.groupby(keys, sort=False, observed=True, dropna=False).indices.items())
        repeated = sum(len(positions) > 1 for _, positions in groups)
        result.notes.append(f"{len(groups):,} entities; {repeated:,} have repeated observations. Entity key: {', '.join(map(str, keys))}.")
        if repeated:
            result.notes.append("Longitudinal structure confirmed by unique entity + sequence keys." if sequence else "Repeated-entity structure present; time order is unconfirmed.")
            result.notes.append("Repeated measures can share entity effects even without serial correlation. Validate by time for future observations and by entity for unseen entities.")
        else:
            result.notes.append("Every entity is a singleton: no within-entity history to test. Check that Identifier roles describe entities rather than unique row IDs.")
    else:
        groups = [("All observations", np.arange(len(frame)))]
        result.notes.append("One ordered series; no entity grouping. Temporal design depends on what the Sequence variable represents." if sequence else "No entity grouping.")
    allowed = {Role.PREDICTOR, Role.TARGET} if use_target else {Role.PREDICTOR}
    excluded = {Role.IDENTIFIER, Role.SEQUENCE, Role.WEIGHTING, Role.GEOMETRY}
    columns = [c for c in frame if roles.roles_for(c) & allowed and not roles.roles_for(c) & excluded
               and not str(c).startswith(Card.SHADOW_PREFIX) and _numeric(frame[c])]
    if not columns:
        result.notes.append("No eligible numeric Predictor or enabled Target variables.")
        return result
    if (len(frame) > MAX_ROWS or len(groups) * len(columns) > MAX_SERIES
            or len(frame) * len(columns) * max(1, int(lags)) > MAX_LAG_PRODUCTS):
        result.notes.append(f"Analysis limit exceeded ({MAX_ROWS:,} rows, {MAX_SERIES:,} entity-variable series or {MAX_LAG_PRODUCTS:,} row-variable-lag products). Reduce lags or select a smaller upstream dataset; no random sampling of time sequences is performed.")
        return result
    rows = []
    irregular = 0
    for entity, positions in groups:
        part = frame.iloc[positions]
        if sequence:
            part = part.sort_values(sequence, kind="stable")
        regular = True
        if sequence and len(part) > 2:
            gaps = part[sequence].diff().iloc[1:]
            if pd.api.types.is_timedelta64_dtype(gaps):
                regular = bool((gaps == gaps.iloc[0]).all())
            else:
                regular = bool(np.allclose(gaps.to_numpy(dtype=float), float(gaps.iloc[0]), rtol=1e-7, atol=0))
            irregular += not regular
        label = str(entity)
        for column in columns:
            x = part[column].to_numpy(dtype=float, na_value=np.nan)
            observed = int(np.isfinite(x).sum())
            imputed = trimmed = 0
            if regular and imputation == "rolling" and observed < len(x):
                x, imputed, trimmed = _rolling_impute(x, window)
            h = min(max(1, int(lags)), len(x) // 5)
            row = dict(zip(COLUMNS, [label, str(column), len(part), observed, imputed,
                                    trimmed, len(x), h, np.nan, np.nan, np.nan, np.nan,
                                    "Exploratory: rolling-imputed" if imputed else "Tested"]))
            if not regular:
                row["Status"] = "Irregular spacing: tests unavailable"
            elif not np.isfinite(x).all():
                row["Status"] = "Missing/nonfinite values: tests unavailable"
            elif len(x) < 20 or observed < 20:
                row["Status"] = "Too short (minimum 20 observed values)"
            elif np.all(x == x[0]):
                row["Status"] = "Constant: tests unavailable"
            else:
                row["Ljung–Box p"], row["Lag 1"] = _ljung_box(x, h)
                centered = x / (float(np.max(np.abs(x))) or 1.)
                squared = (centered - centered.mean()) ** 2
                row["Squared p"], _ = _ljung_box(squared, h)
            rows.append(row)
    result.table = pd.DataFrame(rows, columns=COLUMNS)
    # One family across every available variable, entity and both diagnostics.
    p = result.table[["Ljung–Box p", "Squared p"]].to_numpy()
    valid = np.isfinite(p)
    adjusted = np.full(p.shape, np.nan)
    if valid.any():
        adjusted[valid] = multipletests(p[valid], method="holm")[1]
        result.table["Adjusted p"] = pd.DataFrame(adjusted).min(axis=1).to_numpy()
    if irregular:
        result.notes.append(f"{irregular:,} entities have irregular spacing; their tests are unavailable. Resample to a meaningful regular cadence upstream if appropriate.")
    if result.table["Imputed"].sum():
        result.notes.append("Rolling-imputed results are exploratory: imputation can introduce serial correlation; nominal p-values and Holm adjustment do not account for imputation uncertainty. See Imputed and Leading trimmed counts in the table.")
    result.notes.append("Tests use levels and squared deviations, separately within each entity; missing values are never removed to close gaps. Lags are capped at one fifth of series length.")
    return result


@recordable
def _summary(result, alpha):
    valid = result.table["Adjusted p"].notna()
    count = int((result.table["Adjusted p"] < alpha).sum())
    conclusion = (f"Dependence diagnostics flagged {count} of {int(valid.sum())} tested entity-variable series."
                  if count else "No serial dependence detected in tested series; this does not establish independence."
                  if valid.any() else "No series could be tested.")
    return " ".join([conclusion, *result.notes])


@recordable
def _figure(result, *, alpha=.05, full_screen=False):
    table = _display_table(result.table, alpha).dropna(subset=["Adjusted p"]).sort_values("Adjusted p").head(30)
    if table.empty:
        figure = Card.empty_figure("No testable series")
    else:
        figure = go.Figure(go.Bar(
            x=-np.log10(table["Adjusted p"].clip(lower=1e-300)),
            y=table["Entity"] + " / " + table["Variable"], orientation="h",
            marker_color=table["Finding"].map(FINDING_COLOURS), showlegend=False,
            customdata=table[["Adjusted p", "Imputed", "Finding"]].to_numpy(), hovertemplate="%{y}<br>Nominal adjusted p=%{customdata[0]:.4g}<br>Imputed=%{customdata[1]}<br>%{customdata[2]}<extra></extra>"))
        for finding, colour in FINDING_COLOURS.items():
            if table["Finding"].eq(finding).any():
                figure.add_trace(go.Bar(x=[None], y=[None], name=finding,
                                       marker_color=colour, hoverinfo="skip"))
        figure.update_layout(showlegend=full_screen, legend={"orientation": "h", "y": -0.22})
        figure.add_vline(x=-np.log10(alpha), line_dash="dash")
        figure.update_layout(
            template="plotly_white",
            xaxis_title="−log₁₀ adjusted p (larger = stronger evidence)",
            yaxis={"autorange": "reversed", "type": "category"},
            # title="Up to 30 strongest entity-variable results",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor='#bbd6f8',
            margin={"l": 15, "r": 15, "t": 35, "b": 35},
            # font={"size": 13 if full_screen else 10},
        )
    return figure


def instance():
    this = Card(file=__file__, mutable=False)
    this.long_name = "Observation dependence"
    this.description = "Are observations ordered in time or repeated within entities, and is serial dependence detectable?"
    this.front = lambda: ui.TagList(
        ui.span(id="Serial dependence evidence", class_="text-primary text-center d-block"),
        shinywidgets.output_widget(
            id="Chart", fill=True, 
            guide=this, title="Dependence evidence", position="left",
            text="Small Holm-adjusted p-values flag serial correlation in levels or squared deviations. Orange means flagged, yellow means flagged after imputation (exploratory), and blue means not flagged, not proven independent. Only tested series appear; unavailable series remain grey in the table."
        )
    )
    
    this.back = lambda: ui.TagList(
        ui.output_data_frame(
            id="Table", 
            guide=this, title="Entity and variable diagnostics", position="left",
            text="Each entity is tested separately. Shows actual lags, lag-one autocorrelation, raw test p-values and the minimum globally Holm-adjusted p-value of the two tests. Unavailable tests are explained."
        )
    )

    this.footer = lambda: ui.TagList(
        ui.output_ui(id="Busy"), 
        ui.output_text(id="Status")
    )

    this.settings = lambda: ui.TagList(
        ui.input_select(
            id="Order", label="Observation order", choices={"sequence": "Assigned Sequence role", "row": "Exploratory current row order"}, selected="sequence",
            guide=this, position="left", text="Assign exactly one numeric or datetime Sequence upstream. Row order is exploratory and never establishes that data are temporal."
        ),
        ui.input_checkbox(
            id="Entities", label="Group by assigned Identifier columns", value=True, 
            guide=this, position="left",
            text="All Identifier columns form a composite entity key. Use stable entity IDs, not unique row IDs. Disable only when the data truly form one series."
        ),
        ui.input_checkbox(
            id="Target", label="Include numeric targets", value=True, 
            guide=this, position="left",
            text="Tests numeric Predictors and optionally Targets; identifiers, sequence, weights, geometry and shadow columns are excluded."
        ),
        ui.input_select(
            id="Imputation", label="Missing values", choices={"rolling": "Trailing rolling mean (exploratory)", "none": "No imputation"}, selected="rolling",
            guide=this, position="left", 
            text="Fills nonfinite values from original observed values in the preceding window positions within each entity. Leading gaps are trimmed; gaps with no observed history in the window remain unavailable. Imputation may create autocorrelation, so results are exploratory. Exported data are unchanged."
        ),
        ui.input_slider(
            id="Window", label="Rolling window (preceding positions)", min=1, max=50, value=5, step=1,
            guide=this, position="left", 
            text="Uses only earlier observed values, never future or previously imputed values. Larger windows can fill longer gaps but smooth more. At least 20 genuinely observed values are required."
        ),
        ui.input_slider(
            id="Lags", label="Maximum lag", min=1, max=50, value=10, step=1, 
            guide=this, position="left",
            text="Tests all lags up to this maximum, capped at one fifth of each series length; requires at least 20 regularly spaced complete observations per entity and variable."
        ),
        ui.input_slider(
            id="Alpha", label="Adjusted p-value threshold", min=.005, max=.1, value=.05, step=.005, 
            guide=this, position="left", 
            text="Holm correction covers all available tests in this run. Lower thresholds demand stronger evidence. Repeated exploratory changes are not covered by this correction."
        )
    )

    def server(input, output, session):
        busy = this.busy()
        analyze = _analyze

        @this.reactable(calc=True)
        @this.record_context
        def Incoming():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(source is not None)
            return source

        @this.reactable(calc=True)
        @this.settle(seconds=1)
        @this.record_context
        def Options():
            return (int(input.Lags()), input.Order() == "row", bool(input.Entities()), bool(input.Target()), input.Imputation(), int(input.Window()))

        @busy.track("Checking observation dependence…")
        @this.extended_task
        @this.record_context
        async def Calculate(source, options):
            result = analyze(source, *options) if Module.IS_SHINYLIVE else await asyncio.to_thread(analyze, source, *options)
            return source, options, result

        @this.reactable()
        @this.record_context
        def Start():
            Calculate.cancel()
            Calculate.invoke(Incoming().clone(), Options())

        @this.reactable(calc=True)
        @this.record_context
        def Results():
            try:
                source, options, result = Calculate.result()
            except SilentOperationInProgressException:
                req(False)
            req(source.equals(Incoming()) and options == Options())
            return result

        @output
        @render.ui
        @this.record_context
        def Busy():
            return busy.ui()

        @output
        @render.text
        @this.record_context
        def Status():
            return _summary(Results(), float(input.Alpha()))

        @output
        @render.data_frame
        @this.record_context
        def Table():
            table = _display_table(Results().table, float(input.Alpha()))
            return render.DataTable(table, width="100%", height=None, styles=_row_styles(table))

        @output
        @render_widget
        @this.record_context
        def Chart():
            full = bool(this.isFullScreen())
            try:
                figure = _figure(Results(), alpha=float(input.Alpha()), full_screen=full)
            except SilentException:
                figure = Card.empty_figure("Waiting for data or calculation.")
            widget = go.FigureWidget(figure)
            widget._config = {"displayModeBar": full, "displaylogo": False, "responsive": True}
            return widget

        session.on_ended(Calculate.cancel)
        return Incoming

    this.server = server
    return this


if Module.running_directly(name=__name__):
    from roles import RoleMap
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"time": np.arange(200), "value": np.cumsum(rng.normal(size=200))})
    roles = RoleMap.from_primitive({"sequence": ["time"], "predictor": ["value"]})
    this = instance()
    this._imports.set(proxy_data(_df=frame, _roles=roles))
    this.run()
