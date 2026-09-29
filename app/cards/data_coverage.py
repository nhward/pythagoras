"""Mosaic coverage of intersections, relative to mutual independence.

Pearson residual shading follows the vcd/Friendly mosaic convention (2 and 4).
These are descriptive cutoffs, not multiplicity-adjusted significance tests.
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from html import escape
from itertools import product
from math import prod
from pathlib import Path

if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from code_recording import recordable
from module import Module
from proxy_data import proxy_data
from roles import Role, RoleMap
from selection_restore import SelectionRestore
from shiny import reactive, render, req, ui
from shiny.types import SilentOperationInProgressException
from shinywidgets import render_widget

ROLES = {Role.STRATIFIER, Role.TREATMENT, Role.SENSITIVE}
BANDS = [
    ('Strongly under-represented (r ≤ −4)', '#ce506d'),
    ('Under-represented (−4 < r ≤ −2)', '#e29e83'),
    ('Near expectation (−2 < r < 2)', '#afd0be'),
    ('Over-represented (2 ≤ r < 4)', '#92d4de'),
    ('Strongly over-represented (r ≥ 4)', '#4177ac'),
]


@recordable
def _variables(data, max_levels=15):
    result = []
    for c in data.columns:
        if not (data.role_map.roles_for(c) & ROLES) or str(c).startswith(Card.SHADOW_PREFIX):
            continue
        values = data.frame[c]
        if values.map(lambda v: isinstance(v, (list, dict, set, tuple, np.ndarray))).any():
            continue
        if 2 <= values.nunique(dropna=True) <= max_levels:
            result.append(c)
    return result


@recordable
def _band(residual):
    return 0 if residual <= -4 else 1 if residual <= -2 else 4 if residual >= 4 else 3 if residual >= 2 else 2


@recordable
@dataclass
class Coverage:
    variables: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    observed: np.ndarray = field(default_factory=lambda: np.array([]))
    expected: np.ndarray = field(default_factory=lambda: np.array([]))
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    rows: int = 0
    omitted: int = 0
    zero_weight_rows: int = 0
    weighted: bool = False
    total: float = 0.
    note: str = ''
    error: str = ''


@recordable
def _analyze(data, variables, *, max_levels=15, max_cells=256, missing=False, use_weights=False):
    result = Coverage(variables=list(dict.fromkeys(variables)))
    try:
        if not 2 <= len(result.variables) <= 5:
            raise ValueError('Select between two and five Stratifier, Treatment or Sensitive variables.')
        eligible = _variables(data, max_levels)
        if any(c not in eligible for c in result.variables):
            raise ValueError('Selected variables must have an eligible role and between two and the maximum number of observed levels.')
        weight_columns = list(data.role_map.columns_with_role(Role.WEIGHTING))
        result.weighted = bool(use_weights and weight_columns)
        weights = np.ones(len(data.frame))
        if result.weighted:
            if len(weight_columns) != 1:
                raise ValueError('Assign exactly one Weighting variable.')
            series = data.frame[weight_columns[0]]
            if (not pd.api.types.is_numeric_dtype(series.dtype)
                    or pd.api.types.is_bool_dtype(series.dtype)
                    or pd.api.types.is_complex_dtype(series.dtype)):
                raise ValueError('Observation weights must be numeric.')
            weights = series.to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError('Observation weights must be finite and nonnegative.')
        frame = data.frame[result.variables]
        complete = np.ones(len(frame), dtype=bool) if missing else frame.notna().all(axis=1).to_numpy()
        result.omitted = int((~complete).sum())
        result.zero_weight_rows = int((complete & (weights == 0)).sum())
        included = complete & (weights > 0)
        frame = frame.iloc[np.flatnonzero(included)]
        weights = weights[included]
        result.rows = len(frame)
        result.total = float(weights.sum())
        if result.weighted and (not np.isfinite(result.total) or result.total <= 0):
            raise ValueError('Included observations must have a finite, positive total weight.')
        if not result.rows:
            raise ValueError('No observations have values for every selected variable.')
        codes = []
        for c in result.variables:
            code, levels = pd.factorize(frame[c], sort=False)
            labels = list(map(str, levels))
            if (code < 0).any():
                code[code < 0] = len(labels)
                label = '(Missing)'
                while label in labels:
                    label += ' [missing]'
                labels.append(label)
            # Distinguish values such as integer 1 and string "1" in mixed columns.
            labels = [f'{label} [{i + 1}]' if labels.count(label) > 1 else label for i, label in enumerate(labels)]
            codes.append(code)
            result.levels.append(labels)
        shape = tuple(map(len, result.levels))
        if any(size < 2 for size in shape):
            raise ValueError('Each selected variable needs at least two levels in the analyzed rows.')
        if prod(shape) > max_cells:
            raise ValueError(f'{prod(shape):,} intersections exceed the limit of {max_cells:,}. Select fewer variables or increase the intersection limit.')
        observed = np.zeros(shape, dtype=float if result.weighted else int)
        np.add.at(observed, tuple(codes), weights if result.weighted else 1)
        expected = np.full(shape, result.total)
        for axis, size in enumerate(shape):
            marginal = observed.sum(axis=tuple(i for i in range(len(shape)) if i != axis)) / result.total
            broadcast = [1] * len(shape)
            broadcast[axis] = size
            expected *= marginal.reshape(broadcast)
        result.observed, result.expected = observed, expected
        records = []
        for index in product(*(range(size) for size in shape)):
            actual, reference = observed[index].item(), float(expected[index])
            residual = (actual - reference) / np.sqrt(reference)
            record = {f'{c} (level)': result.levels[i][index[i]] for i, c in enumerate(result.variables)}
            record.update({'Observed': actual, 'Expected': reference,
                'Observed %': 100 * actual / result.total, 'Expected %': 100 * reference / result.total,
                'Observed/expected': actual / reference, 'Shortfall': max(0., reference - actual),
                'Pearson residual': residual, 'Coverage': BANDS[_band(residual)][0].split(' (')[0],
                'Empty': actual == 0, 'Expected < 5': reference < 5})
            records.append(record)
        result.table = pd.DataFrame(records).sort_values(['Pearson residual', 'Observed'], kind='stable').reset_index(drop=True)
        if result.weighted:
            result.note = 'Values are sums of weights. Residual shading and the < 5 flag depend on weight scale; interpret descriptively.'
        else:
            if weight_columns:
                result.note += ' The assigned weighting column was not applied.'
            elif use_weights:
                result.note += ' No Weighting role assigned; using counts.'
    except (ValueError, TypeError) as error:
        result.error = str(error)
    return result


@recordable
def _rectangles(masses):
    """Alternate horizontal/vertical splits; positive tile area is mass/total."""
    result = {}

    def split(prefix, bounds):
        depth = len(prefix)
        if depth == masses.ndim:
            result[prefix] = bounds
            return
        totals = np.array([masses[prefix + (i,)].sum() for i in range(masses.shape[depth])], dtype=float)
        # Empty branches remain zero-area but get distinct positions where possible.
        fractions = totals / totals.sum() if totals.sum() else np.full(len(totals), 1 / len(totals))
        start = 0.
        x0, y0, x1, y1 = bounds
        for i, fraction in enumerate(fractions):
            stop = start + fraction
            child = (x0 + start*(x1-x0), y0, x0 + stop*(x1-x0), y1) if depth % 2 == 0 else (x0, y0 + start*(y1-y0), x1, y0 + stop*(y1-y0))
            split(prefix + (i,), child)
            start = stop
    split((), (0., 0., 1., 1.))
    return result


@recordable
def _figure(result, *, area='observed', shade=True, labels=True, full_screen=False):
    if result.error:
        return Card.empty_figure(result.error)
    masses = result.expected if area == 'expected' else result.observed
    figure = go.Figure()
    for index, (x0, y0, x1, y1) in _rectangles(masses).items():
        actual, expected = result.observed[index].item(), float(result.expected[index])
        actual_text = f"{actual:,.4g}" if result.weighted else f"{actual:,}"
        residual = (actual - expected) / np.sqrt(expected)
        color = BANDS[_band(residual)][1] if shade else '#dedede'
        names = [f'{escape(str(c))}: {escape(result.levels[i][index[i]])}' for i, c in enumerate(result.variables)]
        hover = '<br>'.join(names) + f'<br>Observed: {actual_text}<br>Expected: {expected:.4f}<br>Observed/expected: {actual/expected:.4f}<br>Pearson residual: {residual:.4f}'
        if result.weighted:
            hover += '<br>Observed and expected are sums of weights'
        if expected < 5:
            hover += '<br>Small expected ' + ('weight sum' if result.weighted else 'count') + ': interpret cautiously'
        if (x1-x0)*(y1-y0) > 0:
            figure.add_trace(go.Scatter(x=[x0,x1,x1,x0,x0], y=[y0,y0,y1,y1,y0],
                mode='lines', fill='toself', fillcolor=color, line={'color':'white','width':1.5},
                hoveron='fills', text=hover, hoverinfo='text', hovertemplate=hover+'<extra></extra>', showlegend=False))
            if labels and (x1-x0) > .09 and (y1-y0) > .07:
                figure.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2,
                    text='<br>'.join(escape(result.levels[i][index[i]]) for i in range(len(index)))+f'<br>{"Σw" if result.weighted else "n"}={actual_text}',
                    showarrow=False, font={'size':11, 'color':'white' if shade and _band(residual) in (0,4) else '#222'})
        else:
            figure.add_trace(go.Scatter(x=[(x0+x1)/2], y=[(y0+y1)/2], mode='markers',
                marker={'symbol':'diamond-open','size':9,'color':color,'line':{'width':2}},
                hovertemplate=hover+'<br>Zero-area intersection; use Expected areas to separate empty cells<extra></extra>', showlegend=False))
    if shade:
        for name, color in BANDS:
            figure.add_trace(go.Scatter(x=[None],y=[None],mode='markers',marker={'color':color,'size':12,'symbol':'square'},name=name,showlegend=True))
    figure.update_layout(
        template='plotly_white', 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
        margin={'l':65 if full_screen else 12,'r':12,'b':85 if full_screen else 12,'t':48},
        title={
            'text':f'Mosaic chart: Area ∝ {area} ' + ('weighted counts' if result.weighted else 'counts'),
            'font':{'size':17},
            'x': 0.5,
            'xanchor': "center"
        },
        # Later variables subdivide earlier tiles; there is no common category tick scale across all branches, so label the split variables instead.
        xaxis={
            'range':[-.015,1.015], 
            'visible':full_screen,
            'title':{'text':' → '.join(escape(str(c)) for c in result.variables[::2]) if full_screen else None, 'standoff':12},
            'showticklabels':False, 
            'ticks':'', 
            'showgrid':False, 
            'zeroline':False, 
            'showline':False, 
            'automargin':True
        },
        yaxis={
            'range':[-.015,1.015], 
            'visible':full_screen,
            'title':{'text':' → '.join(escape(str(c)) for c in result.variables[1::2]) if full_screen else None, 'standoff':12},
            'showticklabels':False, 
            'ticks':'', 
            'showgrid':False, 
            'zeroline':False, 
            'showline':False, 
            'automargin':True
        },
        showlegend=bool(full_screen and shade), 
        legend={
            'orientation':'h',
            'y':-.15,
            'x':0,
            'itemclick':False,
            'itemdoubleclick':False
        },
        modebar={'orientation':'v'}
    )
    return figure


def instance():
    this = Card(file=__file__, mutable=False)
    this.long_name = 'Data coverage'
    this.description = 'Find under- and over-represented intersections of strata, treatments and sensitive groups.'
    this.defer_configuration_input('Variables')

    this.front = lambda: shinywidgets.output_widget(
        id='Mosaic', fill=True, 
        guide=this, title='Coverage mosaic', position='left',
        text='Each tile is an intersection of the selected levels. Red indicates fewer observations than expected under mutual independence; blue indicates more; grey is near expectation. Splits alternate direction in selection order. Hover for levels, counts (or weight sums when enabled) and residuals. Empty cells have zero area in Observed counts mode: use Expected counts to expose them. Full screen shows the legend and axis titles naming the horizontal and vertical split variables, with arrows indicating successive subdivisions.'
    )

    this.back = lambda: ui.TagList(
        ui.span("Coverage table", class_="text-primary text-center d-block"),
        ui.output_data_frame('Table')
    )

    this.footer = lambda: ui.TagList(ui.output_ui('Busy'), ui.output_ui('Status'))

    this.settings = lambda: ui.TagList(
        ui.input_selectize(
            id='Variables', label='Variables', choices=[], selected=[], multiple=True, options={'plugins':['remove_button']},
            guide=this, position='left',
            text='Select two to five Stratifier, Treatment or Sensitive variables. Initially selects the first two available. Selection order controls mosaic splits. Numeric values are treated as discrete levels. Predictors and targets are excluded.'
        ),
        ui.input_slider(
            id='MaxLevels', label='Maximum levels per variable', min=2, max=50, value=15, step=1,
            guide=this, position='left',
            text='Offers variables with two to this many distinct nonmissing values. No levels are merged. A missing level, when enabled, is additional.'
        ),
        ui.input_slider(
            id='MaxCells', label='Maximum intersections', min=16, max=512, value=256, step=16,
            guide=this, position='left',
            text='Caps the product of level counts, including empty intersections. Excessive tables show an explanation rather than dropping groups. Smaller selections are easier to inspect.'
        ),
        ui.input_radio_buttons(
            id='Area', label='Tile areas', choices={'observed':'Observed counts','expected':'Expected counts'}, selected='observed',
            guide=this, position='left',
            text='Observed counts gives the conventional mosaic: tile area is proportional to count. Empty cells appear as boundary diamonds, which may overlap. Expected counts sizes tiles by independence expectations, making empty and severely under-represented intersections visible. Colour and table values stay the same. When weighting is active, both options use sums of weights instead of counts.'
        ),
        ui.input_checkbox(
            id='Shade', label='Colour by coverage', value=True,
            guide=this, position='left',
            text='Pearson residual = (observed − expected) / square root of expected. Red bands are at −2 and −4; blue at +2 and +4. These are descriptive thresholds, not adjusted significance tests. Expected counts below five are flagged in the table. Independence is a reference, not an assumption that every combination must exist.'
        ),
        ui.input_checkbox(
            id='Missing', label='Include missing values as a level', value=False,
            guide=this, position='left',
            text='Otherwise omit rows missing any selected variable and report how many. All margins and expectations use the same included rows. Missing values in unrelated columns do not exclude observations. Unobserved declared category levels are not included.'
        ),
        ui.input_checkbox(
            id='Labels', label='Label larger tiles', value=True,
            guide=this, position='left',
            text='Show the intersection levels and count where there is room. Hover and the reverse-side table identify all intersections.'
        ),
        ui.input_checkbox(
            id="UseWeights", label="Use assigned observation weighting",
            guide=this, position='left', text= "Sum assigned observation weights for all observed totals, margins, expectations, percentages, residuals and tile areas. Without a Weighting role, use counts. Weights must be finite, numeric and nonnegative; zero-weight rows are omitted. Weights are not normalized, so residual shading depends on their scale."
        ),
    )

    def server(input, output, session):
        selection = SelectionRestore(this.restored_configuration_input('Variables'))
        busy = this.busy()

        @reactive.effect
        @this.record_context
        def ObserveSelection():
            selection.observe(input.Variables() or [])

        @this.reactable(calc=True)
        @this.record_context
        def incomingproxy_data():
            try:
                value = this.input_data()
            except SilentOperationInProgressException:
                # This card does not own the upstream task's progress lifecycle.
                # Clear it normally while waiting, avoiding Shiny's persistent
                # output state when hidden/unhidden or refreshed during that task.
                req(False)
            req(value is not None)
            return value

        @this.reactable()
        @this.record_context
        def Choices():
            choices = _variables(incomingproxy_data(), input.MaxLevels())
            with reactive.isolate():
                current = list(input.Variables() or [])
            ui.update_selectize('Variables', choices=choices, selected=selection.resolve(current, choices, choices[:2]))

        @this.reactable(calc=True)
        @this.settle(2)
        @this.record_context
        def Options():
            return {
                'variables':list(input.Variables() or []),
                'max_levels':int(input.MaxLevels()),
                'max_cells':int(input.MaxCells()),
                'missing':bool(input.Missing()),
                'use_weights':bool(input.UseWeights())
            }

        @busy.track('Assessing coverage…')
        @this.extended_task
        @this.record_context
        async def Calculate(data, options):
            result = _analyze(data, **options) if Module.IS_SHINYLIVE else await asyncio.to_thread(_analyze, data, **options)
            return data, options, result

        @this.reactable()
        @this.record_context
        def Start():
            Calculate.cancel()
            Calculate.invoke(incomingproxy_data().clone(), Options())

        @this.reactable(calc=True)
        @this.record_context
        def Analysis():
            data, options, result = Calculate.result()
            req(data.equals(incomingproxy_data()) and options == Options(), cancel_output=True)
            return result

        @output
        @render_widget
        @this.record_context
        def Mosaic():
            full_screen = bool(this.isFullScreen())
            widget = go.FigureWidget(_figure(Analysis(), area=input.Area(), shade=input.Shade(), labels=input.Labels(), full_screen=full_screen))
            widget._config = {'displayModeBar':full_screen, 'displaylogo':False}
            return widget

        @output
        @render.data_frame
        @this.record_context
        def Table():
            return render.DataTable(Analysis().table.round(4), width='100%', height='auto', filters=False)

        @output
        @render.ui
        @this.record_context
        def Busy():
            return busy.ui()

        @output
        @render.ui
        @this.record_context
        def Status():
            result = Analysis()
            if result.error:
                return result.error
            table = result.table
            text = f'{result.rows:,} observations; {result.omitted:,} incomplete rows omitted; {len(table):,} intersections. '
            if result.weighted:
                text += f'Total weight: {result.total:,.4g}; {result.zero_weight_rows:,} zero-weight rows omitted. '
            elif input.UseWeights():
                text += 'No Weighting role assigned; using counts. '
            under = int((table["Pearson residual"] <= -2).sum())
            over = int((table["Pearson residual"] >= 2).sum())
            if under:
                class_ = "text-warning"
            else:
                class_ = "text-success"
            if this.isFullScreen():
                text += f'{int(table.Empty.sum())} empty; {under} under-represented; {over} over-represented; {int(table["Expected < 5"].sum())} with expected {"weight sum" if result.weighted else "count"} < 5. ' + result.note
            return ui.span(text, class_= class_)

        session.on_ended(Calculate.cancel)

        return this.input_data

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    rows = [('North','Control','A')]*40 + [('North','Treated','B')]*10 + [('South','Control','B')]*5 + [('South','Treated','A')]*45
    frame = pd.DataFrame(rows, columns=['Region','Treatment','Group'])
    roles = RoleMap()
    for column, role in [('Region',Role.STRATIFIER),('Treatment',Role.TREATMENT),('Group',Role.SENSITIVE)]:
        roles.set_roles(column, [role])
    this._imports.set(proxy_data(_df=frame, _roles=roles, _name='Coverage example'))
    this.run()
