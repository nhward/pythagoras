"""A bounded, role-aware matrix of mixed-type pair plots."""
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
from module import Module
from plotly.subplots import make_subplots
from proxy_data import proxy_data
from roles import Role
from selection_restore import SelectionRestore
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget
from var_types import var_kind

MAX_VARIABLES = 12
MAX_LEVELS = 12
MAX_FACETS = 8
MAX_ROWS = 10_000
BINS = 12
NONE = "__no_facet__"
PALETTE = ["#3978a8", "#e6ab02", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#666666", "#a6761d"]
PAIR_COLUMNS = ["X", "Y", "Chart", "Sampled rows", "Plotted rows", "Omitted rows", "Note"]


def _kind(series):
    kind = var_kind(series)
    if kind in {"integer", "decimal", "date-time", "duration"}:
        return "numeric"
    if kind in {"nominal", "ordered", "logical", "text", "object"}:
        return "categorical"
    return kind


def _eligible(source, target=True):
    allowed = {Role.PREDICTOR, Role.TREATMENT} | ({Role.TARGET} if target else set())
    return [c for c in source.frame if not str(c).startswith(Card.SHADOW_PREFIX)
            and source.role_map.roles_for(c) & allowed
            and (target or not source.role_map.has_role(c, Role.TARGET))
            and _kind(source.frame[c]) in {"numeric", "categorical", "cyclic"}]


def _facets(source):
    return [c for c in source.frame if source.role_map.has_role(c, Role.STRATIFIER)
            and not str(c).startswith(Card.SHADOW_PREFIX)
            and var_kind(source.frame[c]) not in {"geometry", "basket", "complex", "unknown"}]


@dataclass
class Column:
    name: str
    kind: str
    values: np.ndarray
    labels: list[str]
    limits: tuple[float, float]
    note: str = ""
    tickvals: list | None = None
    ticktext: list | None = None


@dataclass
class Pairs:
    figure: go.Figure
    table: pd.DataFrame
    sampled: int
    total: int
    notes: list[str]


def _limits(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return (0., 1.)
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        delta = max(abs(lo) * .01, .5)
        return lo - delta, hi + delta
    return lo, hi


def _categorical(series, maximum=MAX_LEVELS):
    """Preserve declared ordering; bound levels without dropping observations."""
    if not all(pd.api.types.is_scalar(value) for value in series):
        return np.full(len(series), np.nan), [], "Non-scalar values cannot be charted."
    if isinstance(series.dtype, pd.CategoricalDtype):
        levels = list(series.cat.categories)
        observed = series.dropna().unique()
        levels = [v for v in levels if v in observed]
        codes = pd.Categorical(series, categories=levels).codes.astype(float)
    else:
        raw, levels = pd.factorize(series, sort=False)
        codes = raw.astype(float)
        levels = list(levels)
    codes[codes < 0] = np.nan
    labels = [str(v) for v in levels]
    note = ""
    if len(levels) > maximum:
        finite = np.isfinite(codes)
        if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
            # Adjacent declared levels stay adjacent after aggregation.
            blocks = np.array_split(np.arange(len(levels)), maximum)
            mapping = {int(v): i for i, block in enumerate(blocks) for v in block}
            labels = [str(levels[b[0]]) if len(b) == 1 else f"{levels[b[0]]} … {levels[b[-1]]}" for b in blocks]
            codes[finite] = [mapping[int(v)] for v in codes[finite]]
            note = f"{len(levels)} ordered levels grouped into {maximum} adjacent bands."
        else:
            counts = np.bincount(codes[finite].astype(int), minlength=len(levels))
            kept = sorted(np.argsort(-counts, kind="stable")[:maximum - 1])
            mapping = {int(v): i for i, v in enumerate(kept)}
            codes[finite] = [mapping.get(int(v), maximum - 1) for v in codes[finite]]
            other = "Other (pooled)"
            while other in labels:
                other += "*"
            labels = [str(levels[i]) for i in kept] + [other]
            note = f"{len(levels)} levels reduced to {maximum - 1} most frequent plus Other."
    return codes, labels, note


def _column(name, series):
    kind = _kind(series)
    labels, note, ticks, text = [], "", None, None
    if kind == "cyclic":
        values = series.cyclic.codes().to_numpy(dtype=float, na_value=np.nan) * (360. / series.cyclic.period)
        labels = [str(v) for v in series.dtype.categories] if series.dtype.is_categorical else []
        ticks = (np.arange(len(labels)) * 360. / len(labels)).tolist() if labels else [0, 90, 180, 270]
        text = labels or [f"{v * series.cyclic.period / 360:g}" for v in ticks]
        limits = (0., 360.)
    elif kind == "categorical":
        values, labels, note = _categorical(series)
        limits = (-.5, max(.5, len(labels) - .5))
        ticks, text = list(range(len(labels))), labels
    else:
        semantic = var_kind(series)
        if semantic == "date-time":
            # Numeric elapsed-day coordinates, date-formatted tick labels.
            valid = series.dropna()
            origin = valid.min() if len(valid) else pd.Timestamp("1970-01-01")
            values = (series - origin).dt.total_seconds().to_numpy(dtype=float, na_value=np.nan) / 86400.
            note = f"Datetime shown on a numeric days-since-{origin} scale."
            limits = _limits(values)
            ticks = np.linspace(*limits, 3).tolist()
            text = [(origin + pd.Timedelta(days=float(v))).strftime("%Y-%m-%d %H:%M") for v in ticks]
        elif semantic == "duration":
            values = series.dt.total_seconds().to_numpy(dtype=float, na_value=np.nan)
            note = "Duration measured in seconds."
            limits = _limits(values)
        else:
            values = series.to_numpy(dtype=float, na_value=np.nan, copy=True)
            limits = _limits(values)
        values[~np.isfinite(values)] = np.nan
    return Column(str(name), kind, values, labels, limits, note, ticks, text)


def _cells(n, layout):
    """Exactly one orientation of every unordered pair, no symmetric repeats."""
    for row in range(1, n):
        for col in range(row):
            yield (col, row) if layout == "checker" and (row + col) % 2 == 0 else (row, col)


def _chart_type(x, y):
    if "cyclic" in (x.kind, y.kind) and x.kind != y.kind:
        return "Polar scatter" if "numeric" in (x.kind, y.kind) else "Polar bars"
    if x.kind == y.kind == "categorical":
        return "Mosaic"
    if x.kind == "categorical":
        return "Horizontal bars"
    if y.kind == "categorical":
        return "Vertical bars"
    return "Scatter"


def _build(source, variables, facet=NONE, limit=1000, target=True, layout="lower"):
    names = list(dict.fromkeys(c for c in variables if c in _eligible(source, target)))[:MAX_VARIABLES]
    total = len(source.frame)
    limit = max(1, min(MAX_ROWS, int(limit)))
    positions = np.sort(np.random.default_rng(1729).choice(total, min(total, limit), replace=False)) if total > limit else np.arange(total)
    frame = source.frame.iloc[positions]
    columns = [_column(c, frame[c]) for c in names]
    notes = [f"{c.name}: {c.note}" for c in columns if c.note]
    if len(variables) > MAX_VARIABLES:
        notes.append("Only the first 12 eligible selected variables are charted.")
    if len(names) < 2 or not len(frame):
        return Pairs(Card.empty_figure("Select at least two eligible variables with observations."), pd.DataFrame(columns=PAIR_COLUMNS), len(frame), total, notes)
    if facet in _facets(source):
        facets, facet_labels, note = _categorical(frame[facet], MAX_FACETS)
        if note:
            notes.append(f"Facet {facet}: {note}")
        if np.isnan(facets).any():
            missing_label = "Missing facet"
            while missing_label in facet_labels:
                missing_label += "*"
            facets[np.isnan(facets)] = len(facet_labels)
            facet_labels.append(missing_label)
    else:
        facet, facets, facet_labels = NONE, np.zeros(len(frame)), ["All observations"]
    n = len(names)
    cells = list(_cells(n, layout))
    specs = [[None for _ in range(n)] for _ in range(n)]
    for row, col in cells:
        specs[row][col] = {"type": "polar" if _chart_type(columns[col], columns[row]).startswith("Polar") else "xy"}
    fig = make_subplots(rows=n, cols=n, specs=specs, horizontal_spacing=.025, vertical_spacing=.025)
    records, shared, legend_seen = [], {}, set()
    axis_titles = {}

    def add(trace, row, col, group):
        label = facet_labels[group]
        trace.update(name=label, legendgroup=str(group), showlegend=facet != NONE and group not in legend_seen,
                     hoverinfo="all")
        legend_seen.add(group)
        fig.add_trace(trace, row=row + 1, col=col + 1)

    def axis(row, col, dimension, column, *, mosaic=False):
        sub = fig.get_subplot(row + 1, col + 1)
        ax = sub.xaxis if dimension == "x" else sub.yaxis
        edge = row == n - 1 if dimension == "x" else col == 0
        axis_titles[ax.plotly_name] = column.name + (" (share)" if mosaic else "")
        settings = {"showticklabels": edge, "showgrid": False, "zeroline": False, "showline": edge,
                        "title": {"text": column.name + (" (share)" if mosaic else "") if edge else "", "font": {"size": 9}}, "tickfont": {"size": 8}}
        if mosaic:
            settings.update(range=[0, 1], tickformat=".0%")
        else:
            settings.update(range=list(column.limits))
            if column.tickvals is not None:
                settings.update(tickmode="array", tickvals=column.tickvals, ticktext=column.ticktext)
            key = (column.name, column.kind, dimension)
            reference = ax.plotly_name.replace("axis", "")
            if key in shared:
                settings["matches"] = shared[key]
            else:
                shared[key] = reference
        ax.update(**settings)

    for row, col in cells:
        x, y = columns[col], columns[row]
        chart = _chart_type(x, y)
        valid = np.isfinite(x.values) & np.isfinite(y.values)
        xv, yv, groups = x.values[valid], y.values[valid], facets[valid].astype(int)
        note = ""
        if chart == "Scatter":
            if x.kind == y.kind == "cyclic":
                note = "Both cycles shown as Cartesian phases; wrap boundaries are equivalent."
            for group in np.unique(groups):
                mask = groups == group
                add(go.Scattergl(x=xv[mask], y=yv[mask], mode="markers",
                    marker={"size": 3, "opacity": .55, "color": PALETTE[group % len(PALETTE)]},
                    hovertemplate=f"{x.name}: %{{x}}<br>{y.name}: %{{y}}<extra>%{{fullData.name}}</extra>"), row, col, group)
            axis(row, col, "x", x)
            axis(row, col, "y", y)
        elif chart in {"Vertical bars", "Horizontal bars", "Polar bars"}:
            polar = chart == "Polar bars"
            numeric = x if x.kind != "categorical" else y
            category = y if y.kind == "categorical" else x
            nums = numeric.values[valid]
            cats = category.values[valid].astype(int)
            edges = np.linspace(*(numeric.limits if not polar else (0., 360.)), BINS + 1)
            mids, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
            counts = np.zeros((len(facet_labels), len(category.labels), BINS))
            for group in np.unique(groups):
                for cat in range(len(category.labels)):
                    counts[group, cat] = np.histogram(nums[(groups == group) & (cats == cat)], edges)[0]
            scale = float(counts.sum(axis=0).max()) if counts.size else 0.
            scale = scale or 1.
            bases = np.repeat(np.arange(len(category.labels)), BINS).astype(float) - (.4 if not polar else 0.)
            for group in np.unique(groups):
                heights = counts[group].ravel() * .8 / scale
                custom = [[category.labels[k], int(counts[group, k, b]), edges[b], edges[b+1]]
                          for k in range(len(category.labels)) for b in range(BINS)]
                hover = f"{category.name}: %{{customdata[0]}}<br>{numeric.name} bin: %{{customdata[2]:.4g}}–%{{customdata[3]:.4g}}<br>Count: %{{customdata[1]}}<extra>%{{fullData.name}}</extra>"
                common = {"base": bases.copy(), "marker_color": PALETTE[group % len(PALETTE)], "customdata": custom, "hovertemplate": hover}
                if polar:
                    trace = go.Barpolar(theta=np.tile(mids, len(category.labels)), r=heights,
                                        width=np.tile(widths * .95, len(category.labels)), **common)
                elif chart == "Vertical bars":
                    trace = go.Bar(x=np.tile(mids, len(category.labels)), y=heights,
                                   width=np.tile(widths * .95, len(category.labels)), **common)
                else:
                    trace = go.Bar(y=np.tile(mids, len(category.labels)), x=heights, orientation="h",
                                   width=np.tile(widths * .95, len(category.labels)), **common)
                add(trace, row, col, group)
                bases += heights
            note = f"Counts in {BINS} bins; band height/width scaled to largest bin count ({int(scale)})."
            if polar:
                polar_axis = fig.get_subplot(row + 1, col + 1)
                polar_axis.update(angularaxis={"showticklabels": row == n-1, "tickmode": "array", "tickvals": numeric.tickvals, "ticktext": numeric.ticktext},
                                  radialaxis={"showticklabels": col == 0, "tickmode": "array", "tickvals": list(range(len(category.labels))), "ticktext": category.labels,
                                              "range": [0, max(1, len(category.labels))]}, barmode="overlay")
                note += f" θ={numeric.name}; radial bands={category.name}."
            else:
                axis(row, col, "x", x)
                axis(row, col, "y", y)
        elif chart == "Mosaic":
            counts = np.zeros((len(x.labels), len(y.labels), len(facet_labels)))
            if len(xv):
                np.add.at(counts, (xv.astype(int), yv.astype(int), groups), 1)
            totals = counts.sum(axis=(1, 2))
            widths = totals / max(1, len(xv))
            left = np.r_[0., np.cumsum(widths)[:-1]]
            bottoms = np.zeros((len(x.labels), len(y.labels)))
            pair_counts = counts.sum(axis=2)
            for i in range(len(x.labels)):
                bottoms[i] = np.r_[0., np.cumsum(pair_counts[i])[:-1]] / max(1, totals[i])
            for group in np.unique(groups):
                heights = counts[:, :, group] / np.maximum(totals[:, None], 1)
                xx, ww = np.repeat(left + widths/2, len(y.labels)), np.repeat(widths, len(y.labels))
                custom = [[x.labels[i], y.labels[j], int(counts[i, j, group])] for i in range(len(x.labels)) for j in range(len(y.labels))]
                colours = PALETTE[group % len(PALETTE)] if facet != NONE else [PALETTE[j % len(PALETTE)] for _ in x.labels for j in range(len(y.labels))]
                add(go.Bar(x=xx, y=heights.ravel(), width=ww, base=bottoms.ravel().copy(),
                    marker={"color": colours, "line": {"color": "white", "width": .6}}, customdata=custom,
                    hovertemplate=f"{x.name}: %{{customdata[0]}}<br>{y.name}: %{{customdata[1]}}<br>Count: %{{customdata[2]}}<extra>%{{fullData.name}}</extra>"), row, col, group)
                bottoms += heights
            axis(row, col, "x", x, mosaic=True)
            axis(row, col, "y", y, mosaic=True)
            note = "Tile area = joint fraction of complete pairs; facet segments partition each tile. Axes show cumulative shares."
        else:
            theta = x if x.kind == "cyclic" else y
            radius = y if x.kind == "cyclic" else x
            low = min(0., radius.limits[0])
            for group in np.unique(groups):
                mask = groups == group
                values = radius.values[valid][mask]
                add(go.Scatterpolar(theta=theta.values[valid][mask], r=values - low, mode="markers",
                    marker={"size": 3, "opacity": .55, "color": PALETTE[group % len(PALETTE)]}, customdata=values,
                    hovertemplate=f"{theta.name} phase: %{{theta}}°<br>{radius.name}: %{{customdata}}<extra>%{{fullData.name}}</extra>"), row, col, group)
            ticks = np.linspace(radius.limits[0], radius.limits[1], 3)
            radial_ticks = ticks - low
            fig.get_subplot(row + 1, col + 1).update(
                angularaxis={"showticklabels": row == n-1, "tickmode": "array", "tickvals": theta.tickvals, "ticktext": theta.ticktext},
                radialaxis={"showticklabels": col == 0, "range": [0, radius.limits[1] - low], "tickmode": "array", "tickvals": radial_ticks, "ticktext": [f"{v:.3g}" for v in ticks]})
            note = f"θ={theta.name}; r={radius.name}. Radial offset={-low:g} (labels show original values)."
        records.append(dict(zip(PAIR_COLUMNS, [x.name, y.name, chart, len(frame), int(valid.sum()), int((~valid).sum()), note])))
    # Diagonal labels identify rows/columns even where outer-edge cells are blank.
    spacing = .025
    size = (1 - spacing * (n - 1)) / n
    for i, name in enumerate(names):
        fig.add_annotation(x=i*(size+spacing)+size/2, y=1-i*(size+spacing)-size/2,
                           xref="paper", yref="paper", text=str(name), showarrow=False, font={"size": 10})
    fig.update_layout(template="plotly_white", barmode="overlay", bargap=0,
                      meta={"pairs_layout": layout, "axis_titles": axis_titles},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#bbd6f8",
                      margin={"l": 70, "r": 15, "t": 15, "b": 65},
                      legend={"title": {"text": str(facet) if facet != NONE else ""},
                              "orientation": "h", "y": -.12}, modebar={"orientation": "v"})
    # Hide inner polar axis decorations as well as labels.
    for name in fig.layout:
        if name.startswith("polar"):
            fig.layout[name].angularaxis.update(showgrid=False, showline=False)
            fig.layout[name].radialaxis.update(showgrid=False, showline=False)
    return Pairs(fig, pd.DataFrame(records, columns=PAIR_COLUMNS), len(frame), total, notes)


def _display(result, full):
    fig = go.Figure(result.figure)
    fig.update_layout(showlegend=full, hovermode="closest" if full else False,
                      font={"size": 12 if full else 9})
    if full and (fig.layout.meta or {}).get("pairs_layout") == "checker":
        for name, title in fig.layout.meta["axis_titles"].items():
            fig.layout[name].update(showticklabels=True, showline=True,
                                    title_text=title, ticks="outside")
        for name in fig.layout:
            if name.startswith("polar"):
                fig.layout[name].angularaxis.update(showticklabels=True, showline=True, ticks="outside")
                fig.layout[name].radialaxis.update(showticklabels=True, showline=True, ticks="outside")
    if not full:
        for trace in fig.data:
            trace.update(hoverinfo="skip", hovertemplate=None)
    fig.update_xaxes(fixedrange=not full)
    fig.update_yaxes(fixedrange=not full)
    return fig


def instance():
    this = Card(file=__file__, mutable=False)
    this.long_name = "Variable pairs"
    this.description = "Explore pairs of original variables using charts suited to their semantic types."
    
    this.front = lambda: ui.TagList(
        shinywidgets.output_widget(
            id="Chart", fill=True, 
            guide=this, title="Pairs grid", position="left",
            text="One chart per variable pair. Diagonal labels identify variables. Mixed bars show binned counts within category bands; mosaics show joint proportions. Hover and legends appear only in full screen."
        )
    )
    
    this.footer = lambda: ui.TagList(
        ui.output_ui(id="Busy"), 
        ui.output_text(id="Status")
    )
    
    this.settings = lambda: ui.TagList(
        ui.input_checkbox(
            id="Target", label="Include targets", value=True, 
            guide=this, position="left", 
            text="Only Predictor, Treatment and optionally Target roles are eligible. Code, geometry, shadow and unsupported structured types are excluded. Use before encoding."
        ),
        ui.input_selectize(
            id="Variables", label="Variables (maximum 12)", choices=[], multiple=True, options={"maxItems": MAX_VARIABLES}, 
            guide=this, position="left", text="Select 2–12 variables. Selection order determines rows and columns; ordered-category order is retained."
        ),
        ui.input_select(
            id="Facet", label="Colour by stratifier", choices={NONE: "None"}, selected=NONE, 
            guide=this, position="left", text="One assigned Stratifier colours points and partitions stacked bars or mosaic tiles. More than eight levels are pooled; missing facet values form a separate group."
        ),
        ui.input_select(
            id="Layout", label="Grid layout", choices={"lower": "Lower triangle", "checker": "Checkerboard (alternating sides)"}, selected="lower", 
            guide=this, position="left", text="Both layouts show every pair once. Alternating sides places successive diagonals in opposite halves, leaving their symmetric counterparts empty."
        ),
        ui.input_slider(
            id="Limit", label="Maximum observations", min=2, max=4, value=3, step=1, ticks=True, pre="10^", 
            guide=this, position="left", text="A fixed-seed random sample of up to 10^n rows is shared by all panels. Pairwise missing/nonfinite values are then omitted."
        )
    )

    def server(input, output, session):
        busy = this.busy()
        selection = SelectionRestore(this.restored_configuration_input("Variables"))
        saved = this.restored_configuration_input("Facet")
        facet_selection = SelectionRestore(None if saved is None else [] if saved == NONE else [saved])
        build = this.record_code(_build)

        @reactive.effect
        def ObserveSelections():
            selection.observe(input.Variables() or [])
            value = input.Facet()
            facet_selection.observe([value] if value and value != NONE else [])

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
            eligible = _eligible(source, bool(input.Target()))
            facets = _facets(source)
            with reactive.isolate():
                selected, facet = list(input.Variables() or []), input.Facet()
            chosen = selection.resolve(selected, eligible, eligible[:6])[:MAX_VARIABLES]
            selected_facet = facet_selection.resolve([facet] if facet and facet != NONE else [], facets, [])
            ui.update_selectize("Variables", choices=eligible, selected=chosen)
            ui.update_select("Facet", choices={NONE: "None"} | {c: c for c in facets}, selected=selected_facet[0] if selected_facet else NONE)

        @this.reactable(calc=True)
        @this.settle(seconds=1)
        def Options():
            return (tuple(input.Variables() or [])[:MAX_VARIABLES], input.Facet() or NONE,
                    int(10 ** input.Limit()), bool(input.Target()), input.Layout())

        @busy.track("Drawing variable pairs…")
        @this.extended_task
        async def Calculate(source, options):
            result = build(source, *options) if Module.IS_SHINYLIVE else await asyncio.to_thread(build, source, *options)
            return source, options, result

        @this.reactable()
        def Start():
            Calculate.cancel()
            Calculate.invoke(Incoming().clone(), Options())

        @this.reactable(calc=True)
        def Results():
            try:
                source, options, result = Calculate.result()
            except SilentOperationInProgressException:
                req(False)
            req(source.equals(Incoming()) and options == Options())
            return result

        @output
        @render_widget
        def Chart():
            full = bool(this.isFullScreen())
            try:
                fig = _display(Results(), full)
            except SilentException:
                fig = Card.empty_figure("Waiting for data or calculation.")
            widget = go.FigureWidget(fig)
            widget._config = {"displayModeBar": full, "displaylogo": False, "responsive": True}
            return widget

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render.text
        def Status():
            r = Results()
            return f"{r.sampled:,} of {r.total:,} rows sampled; {len(r.table)} pairs. Pairwise missing values omitted. " + " ".join(r.notes)

        session.on_ended(Calculate.cancel)
        return Incoming

    this.server = server
    return this


if Module.running_directly(name=__name__):
    from roles import RoleMap
    rng = np.random.default_rng(1729)
    frame = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200), "group": pd.Categorical(np.tile(["A", "B"], 100))})
    this = instance()
    this._imports.set(proxy_data(_df=frame, _roles=RoleMap.from_primitive({"predictor": list(frame)})))
    this.run()
