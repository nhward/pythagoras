"""Numeric distributions and one-way comparisons across assigned strata."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    # Ensure local modules and packages are resolved from the app directory.
    os.chdir(ROOT)
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

import asyncio
from dataclasses import dataclass, field
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from module import Module
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from proxy_data import proxy_data as pxd
from roles import Role
from scipy.stats import f as f_distribution
from selection_restore import SelectionRestore
from shiny import reactive, render, req, ui
from shinywidgets import render_widget


def _variables(data, include_target=False):
    roles = {Role.PREDICTOR, Role.TARGET} if include_target else {Role.PREDICTOR}
    return [c for c in data.columns if data.role_map.roles_for(c) & roles
            and not str(c).startswith(Card.SHADOW_PREFIX)
            and pd.api.types.is_numeric_dtype(data.frame[c].dtype)
            and not pd.api.types.is_bool_dtype(data.frame[c].dtype)
            and not pd.api.types.is_complex_dtype(data.frame[c].dtype)]


def _stratifiers(data):
    return [c for c in data.columns if Role.STRATIFIER in data.role_map.roles_for(c)
            and not data.frame[c].map(lambda x: isinstance(x, (list, dict, set, tuple, np.ndarray))).any()]


def _sample(codes, limit):
    """Proportional deterministic sample, retaining at least one row per stratum."""
    if len(codes) <= limit:
        return np.arange(len(codes))
    groups = [np.flatnonzero(codes == g) for g in np.unique(codes)]
    counts = np.array([len(g) for g in groups])
    if len(groups) > limit:
        raise ValueError('The observation cap must be at least the number of strata.')
    remaining = limit - len(groups)
    fractions = remaining * (counts - 1) / (counts - 1).sum()
    extra = np.floor(fractions).astype(int)
    for i in np.argsort(-(fractions - extra), kind='stable')[:remaining - extra.sum()]:
        extra[i] += 1
    rng = np.random.default_rng(2025)
    return np.sort(np.concatenate([rng.choice(g, size=int(n + 1), replace=False) for g, n in zip(groups, extra)]))


def _anova(groups, method):
    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)
    n = sum(map(len, groups))
    result = {'Groups':k, 'N':n, 'DF between':np.nan, 'DF denominator':np.nan,
              'F':np.nan, 'p':np.nan, 'Eta squared':np.nan, 'Status':''}
    if k < 2 or any(len(g) < 2 for g in groups):
        result['Status'] = 'At least two finite observations in every stratum are required.'
        return result
    # Rescale before sums of squares to avoid overflow without changing F or eta².
    scale = float(np.max(np.abs(np.concatenate(groups)))) or 1.
    groups = [g / scale for g in groups]
    counts = np.array([len(g) for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    overall = np.average(means, weights=counts)
    between = np.sum(counts * (means - overall)**2)
    within = np.sum((counts - 1) * variances)
    if between + within == 0:
        result['Status'] = 'All observed values are identical; no variance to compare.'
        return result
    result['Eta squared'] = between / (between + within)
    result['DF between'] = k - 1
    if method == 'welch':
        if np.any(variances <= 0):
            result['Status'] = 'Welch ANOVA requires positive variance in every stratum.'
            return result
        weights = counts / variances
        proportions = weights / weights.sum()
        correction = np.sum((1 - proportions)**2 / (counts - 1))
        center = np.sum(proportions * means)
        statistic = np.sum(weights * (means - center)**2) / (k - 1)
        statistic /= 1 + 2 * (k - 2) * correction / (k*k - 1)
        df = (k*k - 1) / (3 * correction)
    elif method == 'ordinary':
        df = n - k
        statistic = np.inf if within == 0 else (between / (k - 1)) / (within / df)
    else:
        raise ValueError('Unknown ANOVA method')
    result.update({'F':float(statistic), 'DF denominator':float(df),
                   'p':float(f_distribution.sf(statistic, k - 1, df)),
                   'Status':'Exploratory comparison; check independence and distribution assumptions.'})
    return result


def _adjust_p(values):
    result = np.full(len(values), np.nan)
    positions = np.flatnonzero(np.isfinite(values))
    order = positions[np.argsort(np.asarray(values)[positions])]
    if len(order):
        adjusted = np.asarray(values)[order] * len(order) / np.arange(1, len(order) + 1)
        result[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


@dataclass
class Distributions:
    variables: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    values: dict = field(default_factory=dict)
    hover: dict = field(default_factory=dict)
    anova: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    rows: int = 0
    eligible: int = 0
    missing_strata: int = 0
    normalization: str = 'none'
    note: str = ''
    error: str = ''


def _analyze(data, variables, stratifier='', *, include_target=False, limit=5000,
             normalization='none', method='ordinary', max_strata=12):
    result = Distributions(normalization=normalization)
    try:
        result.variables = [c for c in dict.fromkeys(variables) if c in _variables(data, include_target)]
        if not result.variables:
            raise ValueError('Select at least one numeric Predictor or enable numeric Targets.')
        frame = data.frame
        if stratifier:
            if stratifier not in _stratifiers(data):
                raise ValueError('Choose an available Stratifier-role column.')
            valid = frame[stratifier].notna().to_numpy()
            result.missing_strata = int((~valid).sum())
            positions = np.flatnonzero(valid)
            codes, levels = pd.factorize(frame[stratifier].iloc[positions], sort=False)
            result.levels = list(map(str, levels))
        else:
            positions = np.arange(len(frame))
            codes = np.zeros(len(frame), dtype=int)
            result.levels = ['All observations']
        if not len(positions):
            raise ValueError('No observations have a recorded stratum.')
        if len(result.levels) > max_strata:
            raise ValueError(f'{len(result.levels)} strata exceed the display limit of {max_strata}; choose another Stratifier or raise the limit.')
        result.eligible = len(positions)
        sampled = _sample(codes, int(limit))
        positions, codes = positions[sampled], codes[sampled]
        result.rows = len(positions)
        identifiers = [c for c in frame if Role.IDENTIFIER in data.role_map.roles_for(c)]
        hover = []
        for position in positions:
            parts = [f'{escape(str(c))}: {escape(str(frame[c].iloc[position]))}' for c in identifiers
                     if pd.api.types.is_scalar(frame[c].iloc[position]) and not pd.isna(frame[c].iloc[position])]
            hover.append('<br>'.join(parts) or f'Row {position + 1}')
        hover = np.array(hover, dtype=object)
        summaries, tests = [], []
        for variable in result.variables:
            raw = frame[variable].iloc[positions].to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(raw)
            plotted = raw.copy()
            if finite.any() and normalization != 'none':
                safe_scale = float(np.max(np.abs(raw[finite]))) or 1.
                safe = raw[finite] / safe_scale
                mean, sd = safe.mean(), safe.std(ddof=1) if finite.sum() > 1 else 0
                if normalization == 'center':
                    plotted[finite] = (safe - mean) * safe_scale
                elif normalization == 'zscore':
                    plotted[finite] = (safe - mean) / sd if sd > 0 else 0
                else:
                    raise ValueError('Unknown normalization')
            groups = []
            for group, level in enumerate(result.levels):
                mask = (codes == group) & finite
                values = raw[mask]
                groups.append(values)
                result.values[variable, group] = plotted[mask]
                result.hover[variable, group] = hover[mask]
                q = np.quantile(values, [.25,.5,.75]) if len(values) else [np.nan]*3
                summaries.append({'Variable':variable, 'Stratum':level, 'N':len(values),
                    'Missing/nonfinite':int(((codes == group) & ~finite).sum()),
                    'Mean':float(values.mean()) if len(values) else np.nan,
                    'SD':float(values.std(ddof=1)) if len(values)>1 else np.nan,
                    'Q1':q[0], 'Median':q[1], 'Q3':q[2]})
            row = {'Variable':variable, 'Stratifier':stratifier or 'None', 'Method':method.title()}
            row.update(_anova(groups, method) if stratifier else {'Status':'Choose a Stratifier for ANOVA.', 'p':np.nan})
            tests.append(row)
        result.anova = pd.DataFrame(tests)
        result.anova['BH adjusted p'] = _adjust_p(result.anova.p.to_numpy())
        result.summary = pd.DataFrame(summaries)
        result.note = 'Plots, summaries and ANOVA use the same sampled rows; equal observation importance.'
        if data.role_map.columns_with_role(Role.WEIGHTING):
            result.note += ' The assigned weighting column is not applied.'
        if stratifier.startswith('cluster_'):
            result.note += ' Cluster-defined strata make p-values descriptive, not independent validation of clustering.'
    except (ValueError, TypeError) as error:
        result.error = str(error)
    return result


def _figure(result, *, kind='violin', points=False, inner_box=True, notches=False, mean=False):
    if result.error:
        return Card.empty_figure(result.error)
    rows, cols = len(result.variables), len(result.levels)
    fig = make_subplots(rows=rows, cols=cols, shared_yaxes='rows',
        horizontal_spacing=min(.04, .5/max(cols,1)), vertical_spacing=min(.08,.5/max(rows,1)),
        column_titles=[escape(s) for s in result.levels])
    for row, variable in enumerate(result.variables,1):
        pooled = np.concatenate([result.values[variable,g] for g in range(cols)])
        low, high = (pooled.min(), pooled.max()) if len(pooled) else (0,1)
        pad = max(float(high-low)*.08, abs(float(low))*.01, .01)
        for group in range(cols):
            values = result.values[variable,group]
            common = {"y": values, "x": [result.levels[group]]*len(values),
                "name": result.levels[group], "legendgroup": str(group), "showlegend": False,
                "marker_color": qualitative.Plotly[group % len(qualitative.Plotly)],
                "marker_opacity": 0.35,
                "customdata": result.hover[variable,group][:,None],
                "hovertemplate": '%{customdata[0]}<br>Value: %{y:.5g}<extra></extra>'}
            if kind == 'violin' and len(np.unique(values)) > 1:
                trace = go.Violin(**common, box_visible=inner_box, meanline_visible=mean,
                    points='all' if points else False, spanmode='hard', scalemode='width', scalegroup=variable, jitter=.2)
            else:
                trace = go.Box(**common, notched=notches if kind=='box' else False,
                    boxmean=mean, boxpoints='all' if points else False, quartilemethod='linear', jitter=.2)
            fig.add_trace(trace,row=row,col=group+1)
            fig.update_xaxes(type='category',showticklabels=False,row=row,col=group+1)
            fig.update_yaxes(range=[low-pad,high+pad],row=row,col=group+1,
                title_text=escape(str(variable)) + (' (z)' if result.normalization=='zscore' else ' (centered)' if result.normalization=='center' else '') if group==0 else None)
            if not len(values):
                axis=(row-1)*cols+group+1
                suffix='' if axis==1 else str(axis)
                fig.add_annotation(text='No finite values',xref=f'x{suffix} domain',yref=f'y{suffix} domain',x=.5,y=.5,showarrow=False)
    fig.update_layout(template='plotly_white',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#e5ecf6',
        margin={'l':50,'r':15,'t':35,'b':15},showlegend=False,modebar={'orientation':'v'})
    return fig


def instance():
    this = Card(file=__file__)
    this.long_name = 'Data strata comparison'
    this.description = 'Compare numeric distributions across strata with violin or box plots and ANOVA.'
    this.defer_configuration_input('Variables')
    this.defer_configuration_input('Stratifier')
    this.front = lambda: shinywidgets.output_widget('Plots',fill=True,guide=this,position='left',title='Strata distributions',
        text='Rows are selected variables; columns are levels of the chosen Stratifier. Within a variable all facets share the same vertical scale. Violin widths show density, not sample size; constant or singleton groups use a box instead. Boxes show quartiles, median, and 1.5-IQR whiskers. Raw points can be shown with Identifier hover. Full screen helps when several variables or strata are selected.')
    this.back = lambda: ui.navset_tab(
        ui.nav_panel('ANOVA',ui.output_data_frame('Anova')),
        ui.nav_panel('Group summaries',ui.output_data_frame('Summaries')))
    this.footer = lambda: ui.TagList(ui.output_ui('Busy'),ui.output_text('Status'))
    def settings():
        return ui.TagList(
            ui.input_selectize(id='Variables',label='Variables',choices=[],selected=[],multiple=True,options={'plugins':['remove_button']},guide=this,position='left',text='Initially selects only the first numeric Predictor. Select more variables or clear all. Numeric Target columns become available when enabled. Boolean, complex, identifier, weighting and shadow columns are excluded.'),
            ui.input_checkbox(id='IncludeTarget',label='Offer numeric Target variables',value=False,guide=this,position='left',text='Add numeric Target-role columns to the variable choices. They are not selected automatically.'),
            ui.input_select(id='Stratifier',label='Facet by Stratifier',choices={'':'None'},selected='',guide=this,position='left',text='Select an assigned Stratifier; initially the first available one is used. None shows overall distributions. Missing stratum labels are omitted. Numeric stratum values are treated as levels, not continuous measurements.'),
            ui.input_radio_buttons(id='Kind',label='Chart type',choices={'violin':'Violin','box':'Box plot'},selected='violin',inline=True,guide=this,position='left',text='Violin shows a smoothed distribution; box plot emphasizes quartiles. Smoothing can suggest detail unsupported by small samples.'),
            ui.input_select(id='Normalization',label='Display normalization',choices={'none':'Original units','center':'Center','zscore':'Center and scale'},selected='none',guide=this,position='left',text='Center and scale across all analyzed strata of each variable, never within strata. This changes only the display. ANOVA and summaries always use original units; constant variables map to zero with scaling.'),
            ui.input_checkbox(id='Points',label='Show observations',value=False,guide=this,position='left',text='Overlay jittered points with Identifier-role values on hover, falling back to original row positions. Points receive equal importance.'),
            ui.input_checkbox(id='InnerBox',label='Box inside violin',value=True,guide=this,position='left',text='Show an internal box in violin mode. Does not affect pure box plots.'),
            ui.input_checkbox(id='Notches',label='Notched box plots',value=False,guide=this,position='left',text='Show approximate median uncertainty notches in box mode only. These are not confidence intervals for means and can extend beyond a small sample box.'),
            ui.input_checkbox(id='Mean',label='Show mean',value=False,guide=this,position='left',text='Show a mean marker or line alongside the distribution. Means may be sensitive to outliers.'),
            ui.input_select(id='Method',label='ANOVA method',choices={'ordinary':'Ordinary one-way ANOVA','welch':'Welch ANOVA'},selected='ordinary',guide=this,position='left',text='Compares stratum means for each selected variable. Ordinary ANOVA assumes equal variances; Welch permits unequal variances. Both assume independent observations and need distribution assumptions for p-values. Each stratum needs at least two finite values; Welch also needs positive variances. Eta squared describes between-stratum variation and is not Welch-adjusted. BH p-values adjust across valid selected-variable tests. Cluster-derived strata are exploratory: small p-values do not validate their clusters.'),
            ui.input_slider(id='MaxStrata',label='Maximum displayed strata',min=2,max=24,value=12,step=1,guide=this,position='left',text='If there are more observed levels, show an explanation rather than silently dropping groups. Increase only if the resulting facet grid remains readable.'),
            ui.input_slider(
                id = "MaxObs", label = "Maximum observations to analyze", min = 3, max = 7, value = 4, ticks = True, pre = "10^",
                guide = this, position = "left", text = "Sets a cap of observations used to chart the distributions.")
)
    this.settings = settings
    def server(input,output,session):
        busy=this.busy()
        variable_selection=SelectionRestore(this.restored_configuration_input('Variables'))
        saved_facet=this.restored_configuration_input('Stratifier')
        facet_selection=SelectionRestore(None if saved_facet is None else ([saved_facet] if saved_facet else []))

        @this.suspendable()
        def Choices():
            data=this.input_data()
            choices=_variables(data,input.IncludeTarget())
            facets=_stratifiers(data)
            with reactive.isolate():
                previous=list(input.Variables() or [])
                facet=input.Stratifier()
            selected=variable_selection.resolve(previous,choices,_variables(data)[:1])
            # Wait for actual strata before choosing the initial default.
            selected_facet=facet_selection.resolve([facet] if facet else [],facets,facets[:1])
            ui.update_selectize('Variables',choices=choices,selected=selected)
            ui.update_select('Stratifier',choices={**{c:c for c in facets}, '':'None'},selected=selected_facet[0] if selected_facet else '')
        
        @this.settle(2)
        @this.suspendable(calc=True)
        def Options():
            return {
                "variables": list(input.Variables() or []),
                "stratifier": input.Stratifier() or '',
                "include_target": input.IncludeTarget(),
                "limit": int(10**input.MaxObs()),
                "normalization": input.Normalization(),
                "method": input.Method(),
                "max_strata": int(input.MaxStrata())
            }

        @busy.track('Comparing strata distributions…')
        @this.extended_task
        async def Calculate(data,options):
            result=_analyze(data,**options) if Module.IS_SHINYLIVE else await asyncio.to_thread(_analyze,data,**options)
            return data,options,result
        
        @this.suspendable()
        def Start():
            Calculate.cancel()
            Calculate.invoke(this.input_data().clone(),Options())
        
        @this.suspendable(calc=True)
        def Analysis():
            data,options,result=Calculate.result()
            req(data.equals(this.input_data()) and options==Options(),cancel_output=True)
            return result
        
        @output
        @render.ui
        def Busy(): 
            return busy.ui()
        
        @output
        @render_widget
        def Plots():
            widget=go.FigureWidget(_figure(Analysis(),kind=input.Kind(),points=input.Points(),inner_box=input.InnerBox(),notches=input.Notches(),mean=input.Mean()))
            widget._config={'displayModeBar':bool(this.isFullScreen()),'displaylogo':False}
            return widget

        @output
        @render.text
        def Status():
            r=Analysis()
            return r.error or f'{r.rows} of {r.eligible} eligible rows; {len(r.variables)} variables; {len(r.levels)} strata. Missing stratum labels: {r.missing_strata}. '+r.note
        
        @output
        @render.data_frame
        def Anova(): 
            return render.DataTable(Analysis().anova.round(4),width='100%',height='auto')
        
        @output
        @render.data_frame
        def Summaries(): 
            return render.DataTable(Analysis().summary.round(5),width='100%',height='auto')
        
        session.on_ended(Calculate.cancel)
        
        return this.input_data
    
    this.server=server
    return this

if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Assmnt.csv")
    px = pxd(_df=df, _name="Ass2")
    this._imports.set(px)
    this.run()
