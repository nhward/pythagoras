"""Role-driven structure overview; dependence evidence never defines data identity."""
from __future__ import annotations

import asyncio
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
from geometry_pandas import is_geometry
from module import Module
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import reactive, render, req, ui
from shiny.types import SilentOperationInProgressException
from shinywidgets import render_widget
from var_types import var_kind

from cards import obs_dependence

PANELS = {"observations": "Observational", "temporal": "Temporal", "spatial": "Spatial", "textual": "Textual"}
FINDINGS = ["Characteristic", "State", "Finding", "Evidence", "Implication"]
NONE = "__none__"


def _roles(source, role):
    return [c for c in source.columns if source.role_map.has_role(c, role)]


def _finding(rows, characteristic, finding, evidence="", implication="", state="Declared and supported"):
    rows.append(dict(zip(FINDINGS, [characteristic, state, str(finding), str(evidence), implication])))


def _thin(values, limit):
    return values[np.unique(np.linspace(0, len(values)-1, min(len(values), limit), dtype=int))] if len(values) else values


@recordable
def _analyze(source, lags=10, use_target=True):
    frame = source.frame
    n = len(frame)
    kinds = {c: var_kind(frame[c]) for c in frame}
    keys = _roles(source, Role.IDENTIFIER)
    sequences = _roles(source, Role.SEQUENCE)
    result = {"findings": {p: [] for p in PANELS}, "conclusions": {}, "kinds": kinds,
              "coverage": [], "intervals": [], "sizes": [], "geometries": {}, "texts": {},
              "sequence": None, "keys": keys}
    obs = result["findings"]["observations"]
    missing = int(frame.isna().sum().sum())
    cells = frame.size
    _finding(obs, "Shape", f"{n:,} observations × {len(frame.columns):,} variables")
    _finding(obs, "Missing cells", f"{missing:,}/{cells:,}" + (f" ({missing/cells:.1%})" if cells else ""))
    for c in frame:
        _finding(obs, str(c), kinds[c], ", ".join(r.value for r in source.role_map.roles_for(c)) or "No assigned role")
    predictors = set(_roles(source, Role.PREDICTOR) + _roles(source, Role.TREATMENT))
    _finding(obs, "Rows per Predictor/Treatment", f"{n/len(predictors):.2f}" if predictors else "Unavailable: no Predictor/Treatment variables",
             implication="Descriptive ratio, not sample-size adequacy.")
    groups = [("All observations", np.arange(n))]
    key_missing = int(frame[keys].isna().any(axis=1).sum()) if keys else 0
    if keys:
        _finding(obs, "Entity key", ", ".join(map(str, keys)), f"{key_missing:,} rows with incomplete keys",
                 "All Identifier columns form a composite key; unique row IDs can conceal repeated entities.")
        try:
            groups = list(frame.groupby(keys, sort=False, observed=True, dropna=False).indices.items())
        except (TypeError, ValueError):
            groups = []
            key_missing = n or 1
            _finding(obs, "Entity grouping", "Unsupported key values", state="Declared but problematic")
        result["sizes"] = [len(pos) for _, pos in groups] if not key_missing else []
    repeated = sum(size > 1 for size in result["sizes"])
    organization = ("No observations to assess" if not n else
                    "Entity structure unresolved: incomplete or unsupported keys" if key_missing else
                    "Observational unit not established" if not keys else
                    "Repeated/grouped observations" if repeated else
                    "One observed row per declared entity; cross-sectional-compatible")
    result["conclusions"]["observations"] = organization
    if keys and not key_missing:
        _finding(obs, "Entity counts", f"{len(groups):,} entities; {repeated:,} repeated; {len(groups)-repeated:,} singletons")
    _finding(obs, "Study design", "Not inferred from column count or absence of Sequence/Geometry",
             implication="One row per entity does not prove independent observations.", state="Not assessed")

    temporal = result["findings"]["temporal"]
    dates = [c for c, kind in kinds.items() if kind == "date-time"]
    _finding(temporal, "Datetime content", ", ".join(map(str, dates)) or "None identified",
             implication="A datetime predictor need not define observation order.",
             state="Candidate only" if dates and not sequences else "Declared and supported" if dates else "Not identified")
    conclusion = "No temporal ordering declared"
    usable = False
    if len(sequences) > 1:
        conclusion = "Ambiguous ordering: multiple Sequence variables"
    elif len(sequences) == 1:
        c = sequences[0]
        values = frame[c]
        numeric = obs_dependence._numeric(values)
        dated = pd.api.types.is_datetime64_any_dtype(values)
        required = list(dict.fromkeys(keys + [c]))
        if not n:
            conclusion = "Sequence declared; no observations to assess"
        elif not (numeric or dated):
            conclusion = "Unsupported Sequence representation"
        elif key_missing or frame[required].isna().any(axis=None):
            conclusion = "Structure unresolved: missing entity or sequence keys"
        elif numeric and not np.isfinite(values.to_numpy(dtype=float)).all():
            conclusion = "Structure unresolved: nonfinite sequence keys"
        else:
            duplicates = int(frame.duplicated(required, keep=False).sum())
            _finding(temporal, "Duplicate key rows", duplicates, ", ".join(map(str, required)),
                     "Ties can represent events or replicates; no aggregation is performed.")
            if duplicates:
                conclusion = "Structure unresolved: duplicate entity–sequence keys"
            else:
                usable = True
                result["sequence"] = c
                conclusion = ("Longitudinal-compatible structure" if keys and repeated and dated else
                              "Repeated entities over ordered positions; time interpretation unconfirmed" if keys and repeated else
                              "One observed row per entity; no within-entity history" if keys else
                              "Time-indexed observations; event log versus single-entity series unresolved" if dated else
                              "Ordered series; time interpretation unconfirmed")
                unordered = 0
                starts, ends, counts = [], [], []
                same_grid = True
                reference = None
                for entity, positions in groups:
                    part = frame.iloc[positions]
                    unordered += not part[c].is_monotonic_increasing
                    ordered = part.sort_values(c, kind="stable")
                    seq = ordered[c].reset_index(drop=True)
                    if reference is None:
                        reference = seq
                    elif same_grid:
                        same_grid = seq.equals(reference)
                    starts.append(seq.iloc[0]); ends.append(seq.iloc[-1]); counts.append(len(seq))
                    gaps = seq.diff().iloc[1:]
                    result["intervals"].extend(gaps.tolist())
                    result["coverage"].append((str(entity), seq.iloc[0], seq.iloc[-1], positions))
                _finding(temporal, "Sequence", str(c), f"{values.min()} to {values.max()}",
                         "Numeric positions do not establish elapsed time." if numeric else "Datetime order does not establish causation.")
                _finding(temporal, "Current order", f"{unordered:,}/{len(groups):,} groups are not monotonic", implication="Only diagnostic copies are sorted.")
                _finding(temporal, "Coverage", f"{min(counts)}–{max(counts)} rows per group", f"Common starts: {len(set(starts)) == 1}; common finishes: {len(set(ends)) == 1}; identical grids: {same_grid}")
                durations = [end-start for start, end in zip(starts, ends)]
                _finding(temporal, "Observed durations", f"{min(durations)} to {max(durations)}")
                if result["intervals"]:
                    _finding(temporal, "Observed intervals", f"{min(result['intervals'])} to {max(result['intervals'])}", implication="Gaps are not missing records without an expected schedule.")
    _finding(temporal, "Structure", conclusion, ", ".join(map(str, sequences)), state="Declared and supported" if usable else "Declared but problematic" if sequences else "Not identified")
    result["conclusions"]["temporal"] = conclusion
    try:
        dep = obs_dependence._analyze(source, lags=max(1, min(50, int(lags))), use_target=use_target, imputation="none")
    except (TypeError, ValueError) as error:
        dep = obs_dependence.Result(notes=[f"Dependence unavailable: {error}"])
    result["dependence"] = dep.table
    tested = dep.table["Adjusted p"].notna()
    flagged = int((dep.table["Adjusted p"] < .05).sum())
    count = int(tested.sum())
    unavailable = len(dep.table)-count
    dependency = (f"Serial dependence detected in {flagged}/{count} tested series; {unavailable} unavailable" if flagged else
                  f"No serial dependence detected in {count} assessed series; independence is not established" if count else
                  "Serial dependence not assessable")
    result["dependency"] = dependency
    _finding(temporal, "Serial dependence", dependency, " ".join(dep.notes),
             "Holm-adjusted 5% tests of levels and squared deviations; no imputation or observation weights. Does not determine temporal identity.",
             state="Declared and supported" if count else "Not assessed")
    if count and not flagged:
        _finding(temporal, "Unavailable series", unavailable, implication="See detailed diagnostics for eligibility failures.")

    spatial = result["findings"]["spatial"]
    declared = _roles(source, Role.GEOMETRY)
    geometry_columns = list(dict.fromkeys(declared + [c for c in frame if is_geometry(frame[c])]))
    for c in geometry_columns:
        series = frame[c]
        if not is_geometry(series):
            _finding(spatial, str(c), "Geometry role with unsupported representation", str(series.dtype), "Convert geometry upstream.", "Declared but problematic")
            continue
        try:
            import geopandas as gpd
            geo = gpd.GeoSeries(series)
            present = ~geo.isna()
            empty = geo.is_empty & present
            valid = geo.is_valid & present & ~empty
            crs = str(geo.crs) if geo.crs else "Unknown"
            bounds = geo[valid].total_bounds if valid.any() else None
            result["geometries"][c] = {"geo": geo, "valid": valid.to_numpy(), "crs": crs, "bounds": bounds}
            _finding(spatial, str(c), f"{int(valid.sum())}/{n} valid nonempty; {int((~present).sum())} missing; {int(empty.sum())} empty; {int((present & ~empty & ~valid).sum())} invalid",
                     f"CRS: {crs}; types: {', '.join(geo[present].geom_type.unique())}; Z present: {bool(geo.has_z.any())}; bounds: {bounds}",
                     "X/Y preview only. Geometry does not establish spatial dependence.",
                     "Declared and supported" if c in declared else "Candidate only")
        except (ImportError, ValueError, TypeError, AttributeError) as error:
            _finding(spatial, str(c), f"Geometry assessment unavailable: {error}", state="Not assessed")
    lat = [c for c in frame if str(c).lower() in ("lat", "latitude") and obs_dependence._numeric(frame[c])]
    lon = [c for c in frame if str(c).lower() in ("lon", "long", "longitude") and obs_dependence._numeric(frame[c])]
    for a in lat:
        for b in lon:
            if frame[a].notna().any() and frame[b].notna().any() and frame[a].dropna().between(-90, 90).all() and frame[b].dropna().between(-180, 180).all():
                _finding(spatial, "Coordinate candidate", f"{a}, {b}", "Names and observed ranges are plausible", "Confirm axes, CRS and Geometry role upstream.", "Candidate only")
    result["conclusions"]["spatial"] = ("Spatial geometry declared" if declared else "Unassigned geometry content" if geometry_columns else "No spatial geometry declared")
    if not spatial:
        _finding(spatial, "Geometry", "None identified", state="Not identified")

    textual = result["findings"]["textual"]
    for c, kind in kinds.items():
        if kind != "text":
            continue
        values = frame[c]
        lengths = np.full(n, np.nan)
        blank = nonstring = 0
        for i, value in enumerate(values):
            if isinstance(value, str):
                blank += not value.strip()
                lengths[i] = len(value)
            elif not pd.isna(value):
                nonstring += 1
        result["texts"][c] = lengths
        finite = lengths[np.isfinite(lengths)]
        _finding(textual, str(c), f"{int(values.isna().sum())} missing; {blank} empty/whitespace; {len(finite)-blank} nonempty; {nonstring} nonstring",
                 f"Unicode code points: min/median/max {np.min(finite):g}/{np.median(finite):g}/{np.max(finite):g}" if len(finite) else "No strings to assess",
                 "Roles: " + ", ".join(r.value for r in source.role_map.roles_for(c)))
    result["conclusions"]["textual"] = f"Textual content in {len(result['texts'])} variables" if result["texts"] else "No Text semantic variables declared"
    _finding(textual, "Language, topics and sentiment", "Not assessed", implication="Short or unrecognized-language strings remain text.", state="Not assessed")
    base = conclusion if sequences else organization
    additions = (["spatial geometry declared"] if declared else []) + (["textual content"] if result["texts"] else [])
    if dates and not sequences:
        additions.append("date attributes; no time ordering declared")
    result["summary"] = "; ".join([base] + additions)
    result["findings"] = {p: pd.DataFrame(rows, columns=FINDINGS) for p, rows in result["findings"].items()}
    return result


def _strata(source, column):
    if column not in _roles(source, Role.STRATIFIER):
        return np.repeat("All observations", len(source.frame))
    series = source.frame[column]
    codes, levels = pd.factorize(series, sort=False)
    labels = [f"{i+1}: {v}" for i, v in enumerate(levels)]
    return np.array(["(Missing)" if code < 0 else labels[code] for code in codes])


@recordable
def _figure(source, result, panel, *, stratifier=NONE, geometry=None, traces=(), limit=200, full=False):
    fig = go.Figure()
    frame = source.frame
    groups = _strata(source, stratifier)
    limit = max(1, min(2000, int(limit)))
    note = ""
    palette = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb"]
    colors = {group: palette[i % len(palette)] for i, group in enumerate(dict.fromkeys(groups))}
    colors["Changing stratum"] = "#777777"
    if panel == "observations":
        kinds = pd.Series(list(result["kinds"].values()), dtype=str).value_counts()
        fig.add_trace(go.Bar(x=kinds.index, y=kinds.values, name="Variables"))
        fig.update_xaxes(title="Semantic variable type")
        fig.update_yaxes(title="Variable count")
        if result["sizes"]:
            # An inset preserves both variable composition and entity multiplicity.
            counts = pd.Series(result["sizes"]).value_counts().sort_index()
            fig.add_trace(go.Bar(x=counts.index, y=counts.values, xaxis="x2", yaxis="y2", name="Entities"))
            fig.update_layout(xaxis={"domain": [0, .46]}, xaxis2={"domain": [.58, 1], "title": "Rows per entity"}, yaxis2={"anchor": "x2", "title": "Entities"})
    elif panel == "temporal":
        coverage = result["coverage"]
        sequence = result["sequence"]
        if not sequence:
            return Card.empty_figure(result["conclusions"][panel])
        if result["keys"] and any(size > 1 for size in result["sizes"]):
            for label, start, end, positions in coverage[:limit]:
                strata = set(groups[positions])
                group = next(iter(strata)) if len(strata) == 1 else "Changing stratum"
                fig.add_trace(go.Scatter(x=[start, end], y=[label, label], mode="lines+markers", name=group, line={"color":colors[group]},
                                        legendgroup=group, showlegend=False, text=[group]*2, hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>"))
                if full:
                    points = _thin(np.sort(frame.iloc[positions][sequence].to_numpy()), 500)
                    fig.add_trace(go.Scatter(x=points, y=[label]*len(points), mode="markers", marker={"size": 3, "color":colors[group]}, name=group, showlegend=False))
            note = f"Coverage: {min(limit, len(coverage))}/{len(coverage)} entities; bars do not imply continuous observation"
            fig.update_yaxes(type="category")
        else:
            ordered = frame.sort_values(sequence, kind="stable")
            eligible = [c for c in traces if c in frame and obs_dependence._numeric(frame[c]) and source.role_map.roles_for(c) & {Role.PREDICTOR, Role.TARGET} and c != sequence and c not in result['keys']][:12]
            plotted = 0
            for c in eligible:
                x = ordered[c].to_numpy(dtype=float, na_value=np.nan)
                finite = np.isfinite(x)
                if finite.sum() < 2 or np.std(x[finite]) == 0:
                    continue
                scale = np.max(np.abs(x[finite])) or 1.
                x = x / scale
                std = np.std(x[finite])
                z = (x-np.mean(x[finite]))/std
                positions = _thin(np.arange(len(x)), limit)
                fig.add_trace(go.Scatter(x=ordered[sequence].iloc[positions], y=z[positions], mode="markers" if len(x) > limit else "lines", name=str(c), connectgaps=False))
                plotted += 1
            if not plotted:
                for group in list(dict.fromkeys(groups))[:20]:
                    fig.add_trace(go.Histogram(x=frame.loc[groups == group, sequence], name=group, nbinsx=30, marker_color=colors[group]))
                note = "Observation counts by sequence bin; up to 20 strata. Constant/unusable traces omitted."
                fig.update_yaxes(title="Observations")
            else:
                note = f"Display-only standardized traces; up to {limit} positions each (markers if thinned); constant traces omitted"
                fig.update_yaxes(title="Standard deviations from mean")
        fig.update_xaxes(title=str(sequence))
    elif panel == "textual":
        for c, lengths in result["texts"].items():
            for group in list(dict.fromkeys(groups))[:20]:
                values = lengths[(groups == group) & np.isfinite(lengths)]
                fig.add_trace(go.Box(y=_thin(values, limit), name=f"{c}: {group}", boxpoints=False, marker_color=colors[group]))
        fig.update_yaxes(title="Text length (Unicode code points)")
        note = f"Up to {limit} strings per variable/stratum displayed; up to 20 strata; findings use all rows"
    elif panel == "spatial":
        geometry = geometry if geometry in result["geometries"] else next(iter(result["geometries"]), None)
        if geometry is not None:
            info = result["geometries"][geometry]
            for group in list(dict.fromkeys(groups))[:20]:
                selected = (groups == group) & info["valid"]
                if not selected.any():
                    continue
                x0, y0, x1, y1 = info["geo"].iloc[np.flatnonzero(selected)].total_bounds
                fig.add_trace(go.Scatter(x=[x0,x1,x1,x0,x0], y=[y0,y0,y1,y1,y0], mode="lines+markers", name=group, line={"color":colors[group]}))
            note = f"{geometry}: X/Y extents, CRS {info['crs']}; up to 20 strata. Longitude bounds may cross the antimeridian."
            fig.update_xaxes(title="X coordinate")
            fig.update_yaxes(title="Y coordinate", scaleanchor="x")
    if not fig.data:
        return Card.empty_figure(result["conclusions"][panel])
    fig.update_layout(
        template="plotly_white", 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="#bbd6f8",
        margin={"l":15,"r":15,"t":0,"b":0}, 
        showlegend=full,
        title={"text":note,"font":{"size":11}}, 
        legend={"orientation":"h"}
    )
    return fig


def instance():
    this = Card(file=__file__, mutable=False)
    this.long_name = "Data structure overview"
    this.description = "Distinguish temporal, spatial and textual content from the organization of observations."
    this.front = lambda: ui.navset_bar(
        *[ui.nav_panel(label, 
        ui.output_ui(id=panel + "Summary"), 
        shinywidgets.output_widget(id = panel + "Chart", fill=True), value=panel) for panel, label in PANELS.items()], 
        title=None,
        id="Panel", 
        selected="temporal",
        navbar_options=ui.navbar_options(
            collapsible=False,
            underline = True,
            bg="lightgrey",
            class_="nominal-navbar",
        )
    )

    this.back = lambda: ui.TagList(
        ui.output_ui("DetailTitle"), 
        ui.output_data_frame(id="Findings", guide = this, title = "Findings", text = "Accummulated findings relating to the the current panel"), 
        ui.output_data_frame(id="Dependence")
    )

    this.footer = lambda: ui.TagList(
        ui.output_ui("Busy"), 
        ui.output_ui(id="Status", guide=this, title="Data structure", position="top", text="Overall conclusion about the dataset structure")
    )

    this.settings = lambda: ui.TagList(
        ui.input_select(id="Stratifier", label="Display stratifier", choices={NONE:"None"}, selected=NONE, guide=this, text="Assigned Stratifier only. Missing strata remain visible; chart grouping does not change findings.", position="left"),
        ui.input_select(id="Geometry", label="Geometry preview", choices={}, guide=this, text="Preview one geometry column's X/Y bounds in its own CRS; no coordinate conversion or online tiles.", position="left"),
        ui.input_selectize(id="Traces", label="Temporal numeric traces", choices=[], multiple=True, options={"maxItems":12}, guide=this, text="Up to 12 numeric Predictor/Target traces for a single series. Standardization affects the display only; constant traces are omitted.", position="left"),
        ui.input_slider(id="Limit", label="Display limit", min=10, max=1000, value=200, step=10, guide=this, text="Entities, strings or trace positions displayed. Full-data findings and dependence tests are unaffected.", position="left"),
        ui.input_slider(id="Lags", label="Maximum dependence lag", min=1, max=50, value=10, guide=this, text="Ljung–Box on levels and squared deviations; capped at one fifth of series length. Holm correction across all tests, threshold 5%, minimum 20 complete regularly spaced values. No imputation.", position="left"),
        ui.input_checkbox(id="Target", label="Include numeric targets", value=True, guide=this, text="Include numeric Target variables alongside Predictors in dependence tests. Roles, data and structural classification stay unchanged.", position="left"))

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

        @this.reactable()
        def Choices():
            source = Incoming()
            choices = {NONE:"None", **{c:str(c) for c in _roles(source, Role.STRATIFIER)}}
            geos = [c for c in source.columns if is_geometry(source.frame[c])]
            traces = [c for c in source.columns if obs_dependence._numeric(source.frame[c]) and source.role_map.roles_for(c) & {Role.PREDICTOR, Role.TARGET} and not source.role_map.roles_for(c) & {Role.SEQUENCE, Role.IDENTIFIER, Role.WEIGHTING, Role.GEOMETRY}]
            with reactive.isolate():
                stratifier, geometry = input.Stratifier(), input.Geometry()
                selected = list(input.Traces() or [])
            ui.update_select("Stratifier", choices=choices, selected=stratifier if stratifier in choices else NONE)
            ui.update_select("Geometry", choices=geos, selected=geometry if geometry in geos else geos[0] if geos else "")
            ui.update_selectize("Traces", choices=traces, selected=[c for c in selected if c in traces][:12])

        @this.reactable(calc=True)
        def Options():
            return int(input.Lags() or 10), bool(input.Target())

        @busy.track("Assessing data structure and dependence…")
        @this.extended_task
        @this.record_context
        async def Calculate(source, options):
            result = _analyze(source, *options) if Module.IS_SHINYLIVE else await asyncio.to_thread(_analyze, source, *options)
            return source, options, result

        @this.reactable()
        def Start():
            Calculate.cancel()
            Calculate.invoke(Incoming().clone(), Options())

        @this.reactable(calc=True)
        def Results():
            source, options, result = Calculate.result()
            req(source.equals(Incoming()) and options == Options(), cancel_output=True)
            return result

        def summary(panel):
            result = Results()
            text = result["conclusions"][panel]
            if text.startswith("No "):
                return None
            if panel == "temporal":
                text += ". " + result["dependency"]
            if panel == "observations":
                frame = Incoming().frame
                text += f". {len(frame):,} rows; {len(frame.columns):,} variables; {int(frame.isna().sum().sum()):,} missing cells."
            return ui.span(text, class_="text-secondary")

        def chart(panel):
            figure = _figure(Incoming(), Results(), panel, stratifier=input.Stratifier(), geometry=input.Geometry(), traces=input.Traces() or (), limit=input.Limit(), full=bool(this.isFullScreen()))
            widget = go.FigureWidget(figure)
            widget._config = {"displayModeBar":bool(this.isFullScreen()), "displaylogo":False}
            return widget

        @output
        @render.ui
        def temporalSummary(): 
            return summary("temporal")
        
        @output
        @render.ui
        def spatialSummary(): 
            return summary("spatial")
        
        @output
        @render.ui
        def observationsSummary(): 
            return summary("observations")
        
        @output
        @render.ui
        def textualSummary(): 
            return summary("textual")
        
        @output
        @render_widget
        def temporalChart(): 
            return chart("temporal")
        
        @output
        @render_widget
        def spatialChart(): 
            return chart("spatial")
        
        @output
        @render_widget
        def observationsChart(): 
            return chart("observations")
        
        @output
        @render_widget
        def textualChart(): 
            return chart("textual")

        @output
        @render.ui
        def DetailTitle(): 
            return ui.strong(PANELS.get(input.Panel(), "Temporal"))
        
        @output
        @render.data_frame
        def Findings(): 
            return render.DataTable(Results()["findings"].get(input.Panel(), Results()["findings"]["temporal"]), width="100%", height="auto")
        
        @output
        @render.data_frame
        def Dependence():
            req(input.Panel() == "temporal")
            return render.DataTable(Results()["dependence"].round(5), width="100%", height="auto")
        
        @output
        @render.ui
        def Busy(): 
            return busy.ui()
        
        @output
        @render.ui
        def Status():
            text = Results()["summary"]
            return ui.span(text, title=text, style="display:block;" + ("white-space:normal;" if this.isFullScreen() else "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"))
        
        session.on_ended(Calculate.cancel)
        
        return this.input_data

    this.server = server
    return this


if Module.running_directly(name=__name__):
    frame = pd.DataFrame({"entity": np.repeat(["A", "B"], 30), "time": list(pd.date_range("2025-01-01", periods=30))*2, "value": np.tile(np.arange(30.), 2)})
    roles = RoleMap.from_primitive({"identifier":["entity"], "sequence":["time"], "predictor":["value"]})
    this = instance()
    this._imports.set(proxy_data(_df=frame, _roles=roles))
    this.run()
