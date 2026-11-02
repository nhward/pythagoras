"""Explore and commit fold-fitted original-variable feature selection."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == '__main__':
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
from selection_analysis import (
    analyze_selection,
    export_selection,
    selection_evidence,
    selection_identity,
    selection_stability,
)
from selection_jobs import run_selection
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget

METHODS = ('Boruta', 'GrootCV')


def _display_table(table):
    """Round floating-point display values to four significant figures."""
    displayed = table.copy()
    for column in displayed.select_dtypes(include='floating'):
        displayed[column] = displayed[column].map(
            lambda value: float(f'{value:.4g}') if pd.notna(value) else value)
    return displayed


@recordable
def selection_figure(result, limit=20):
    figure = go.Figure()
    if result is None or result['error']:
        figure.add_annotation(text='Waiting for selection' if result is None else result['error'],
                              x=.5, y=.5, xref='paper', yref='paper', showarrow=False)
    else:
        model = result['model']
        details = model.details_
        table = model.diagnostics_.sort_values('Median importance', ascending=False).head(limit)
        colors = {'Confirmed':'#27864b', 'Selected':'#27864b', 'Tentative':'#bf8800', 'Rejected':'#c44747'}
        for row in table.to_dict('records'):
            index = model.candidate_columns_.index(row['Variable'])
            observed = details['real'][:, index]
            observed = observed[np.isfinite(observed)]
            figure.add_trace(go.Box(x=observed, name=str(row['Variable']), orientation='h',
                marker_color=colors[row['Decision']], boxpoints=False,
                hovertemplate=f"{row['Variable']} · {row['Decision']}<br>Importance=%{{x:.4g}}<extra></extra>"))
        # Each sample is an observed reference, never a fabricated distribution.
        figure.add_trace(go.Box(x=details['thresholds'], name='Shadow threshold',
            orientation='h', marker_color='#3075b7', boxpoints=False))
        figure.add_trace(go.Box(x=np.concatenate(details['shadows']), name='All shadow importances',
            orientation='h', marker_color='#83b5e1', boxpoints=False))
        if result['method'] == 'GrootCV':
            figure.add_vline(x=details['thresholds'][0], line_dash='dash', line_color='#3075b7')
        figure.update_xaxes(title='Normalized forest gain' if result['method'] == 'Boruta'
                             else 'Normalized mean absolute TreeSHAP')
        figure.update_yaxes(autorange='reversed')
    figure.update_layout(showlegend=False, template='plotly_white', plot_bgcolor="#bbd6f8",
        margin={'l': 8, 'r': 8, 't': 8, 'b': 40}, autosize=True)
    return figure


@recordable
def stability_figure(result, limit=20):
    figure = go.Figure()
    if result is None or result['error']:
        figure.add_annotation(text='Calculating fold-by-fold stability…' if result is None else result['error'],
            x=.5, y=.5, xref='paper', yref='paper', showarrow=False)
    else:
        table = result['summary'].sort_values('Selection frequency', ascending=False).head(limit)
        figure.add_trace(go.Bar(x=table['Selection frequency'], y=table.Variable.astype(str),
            orientation='h', marker_color='#3075b7',
            customdata=table[['Selected folds','Eligible folds','Planned folds']].to_numpy(),
            hovertemplate='%{y}<br>Selected: %{customdata[0]} / %{customdata[1]} eligible folds<br>Planned folds: %{customdata[2]}<extra></extra>'))
        figure.update_xaxes(title='Selection frequency among eligible successful folds', range=[0,1], tickformat='.0%')
        figure.update_yaxes(autorange='reversed')
        figure.add_annotation(text=f"{result['successful']} / {result['planned']} training folds succeeded · diagnostic only",
            x=0, y=1.08, xref='paper', yref='paper', showarrow=False, xanchor='left')
    figure.update_layout(showlegend=False, template='plotly_white', plot_bgcolor='#bbd6f8',
        margin={'l':8,'r':8,'t':35,'b':45}, autosize=True)
    return figure


def recommendation_configuration(state):
    """Migrate saved checkbox selections to the scalar radio input."""
    from copy import deepcopy
    if not isinstance(state, dict):
        return state
    state = deepcopy(state)
    legacy = state.pop('selection', None)
    inputs = state.get('inputs')
    if not isinstance(inputs, dict):
        return state
    choice = inputs.get('Recommendation', legacy.get('method') if isinstance(legacy, dict) else 'None')
    if isinstance(choice, list):
        choice = next((value for value in reversed(choice) if value in ('None', *METHODS)), 'None')
    inputs['Recommendation'] = choice if choice in ('None', *METHODS) else 'None'
    return state


def instance():
    this = Card(file=__file__, mutable=True)
    restore = this.restore_configuration_state
    this.restore_configuration_state = lambda state: restore(recommendation_configuration(state))
    this.long_name = 'Variable selection'
    this.description = 'Learn a subset of variables via shadow selection.'
    this.export_inputs('Recommendation')
    
    this.front = lambda: ui.navset_bar(
        *(ui.nav_panel(method, shinywidgets.output_widget(f'Chart_{method}', fill=True), value=method)
          for method in METHODS), id='SelectionMethod', selected='Boruta', title=None,
        padding=0, 
        fillable=True, 
        navbar_options=ui.navbar_options(
            collapsible=False, underline=True, bg='lightgrey', class_='nominal-navbar'
        )
    )
    
    this.back = lambda: ui.div(
        ui.output_text(id='EvidenceTitle'),
        ui.output_text(id='StabilityStatus'),
        ui.output_data_frame(id='Evidence'),
        ui.tags.details(
            ui.tags.summary(id='Technical evidence'), 
            ui.output_text_verbatim(id='Technical'),
            style='max-height: 180px; overflow: auto; flex-shrink: 0;'
        ),
        ui.tags.details(ui.tags.summary('Fold-by-fold stability'), ui.output_data_frame('StabilityDetails'),
            style='max-height: 250px; overflow: auto; flex-shrink: 0;'),
        class_='var-select-evidence html-fill-container html-fill-item', style='min-height: 0;'
    )
    
    this.footer = lambda: ui.TagList(
        ui.output_ui(id='Busy'),
        ui.input_radio_buttons(id='Recommendation', label=None,
            choices=['None', *METHODS], selected='None', inline=True,
            guide=this, title='Selection recommendation', position='top',
            text='The chosen recommendation is applied automatically when ready. None immediately passes through incoming columns. Chart tabs are independent. Settings changes recompute the selected method; pending, failed or empty recommendations pause downstream export. Training refits the procedure inside each fold.'),
        ui.output_text(id='Status')
    )
    
    this.settings = lambda: ui.TagList(
        ui.span("Boruta settings", class_="text-primary"),
        ui.input_radio_buttons(id='Policy', label='Boruta inclusion', choices={'confirmed':'Confirmed', 'tentative':'Confirmed + tentative'}, selected='confirmed',
            guide=this, position="left", text='Choose whether Boruta retains only confirmed variables or also those still tentative when its iteration budget ends. Tentative means unresolved, not confirmed. This policy does not affect GrootCV.'),
        ui.input_slider(id='Trees', label='Trees per fit (budget)', value=200, min=2, max=2000, step=10,
            guide=this, position="left", text='Tree budget for each model fit in both methods. Boruta fits a random forest each iteration; GrootCV uses this as its boosting-round budget and may stop earlier. Larger budgets increase computation.'),
        ui.input_slider(id='Depth', label='Maximum tree depth', value=5, min=1, max=12, step=1,
            guide=this, position="left", text='Maximum tree depth for both selection engines. Deeper trees can capture more complex relationships and interactions, but may overfit and take longer. This changes the selection models, not the eventual downstream model.'),
        ui.input_slider(id='Iterations', label='Boruta iteration budget', value=100, min=3, max=1000, step=10,
            guide=this, position="left", text='Boruta iteration budget. Each iteration compares original variables with shuffled shadows. The loop may finish earlier when every variable is decided. Increasing the budget can resolve tentative decisions; it increases runtime.'),
        ui.input_slider(id='Alpha', label='Boruta significance level', value=.05, min=.001, max=.5, step=.01,
            guide=this, position="left", text='Significance level used by Boruta with its multiple-testing corrections. Larger values make acceptance or rejection easier; smaller values require stronger evidence. This is not a probability that an individual variable is relevant.'),
        ui.input_slider(id='Percentile', label='Boruta shadow percentile', value=100, min=50, max=100, post="%",
            guide=this, position="left", text='Percentile of shadow importance used as the Boruta reference. At 100%, a variable must beat the strongest shadow in an iteration to score a win. Lowering the percentile relaxes that comparison and can retain more variables.'),
        ui.input_checkbox(id='TwoStep', label='Boruta two-step correction', value=True,
            guide=this, position="left", text='Use Benjamini-Hochberg correction across active variables within each iteration, followed by a Bonferroni correction for repeated testing across iterations. When unchecked, BorutaPy uses its original-style Bonferroni correction across variables.'),
        ui.span("Groot settings", class_="text-primary"),
        ui.input_slider(id='Folds', label='GrootCV inner folds', value=5, min=2, max=20, step=1,
            guide=this, position="left", text='Fallback inner fold count when there is no upstream obs_resample design. An upstream design supplies its fold count, grouping, stratification or temporal windows; splits are rebuilt inside each current training partition. Group counts and initial time windows may shrink to fit, with details reported on the reverse.'),
        ui.input_slider(id='Repeats', label='GrootCV repeats', value=5, min=1, max=20, step=1,
            guide=this, position="left", text='Number of times GrootCV repeats inner resampling. Random/grouped designs can change fold assignments; leave-one-group-out, unshuffled and temporal designs retain their boundaries while shadows change. More repeats increase computation. Actual fit counts appear on the reverse.'),
        ui.input_slider(id='Cutoff', label='GrootCV shadow divisor', value=1., min=0.1, max=10., step=0.1,
            guide=this, position="left", text='Divisor applied to the GrootCV shadow reference. A larger divisor lowers the selection threshold and can retain more variables; a smaller divisor raises it. This is not a requested number or proportion of variables to keep.'),
        ui.span("General settings", class_="text-primary"),
        ui.input_checkbox(id='Stability', label='Explore fold-by-fold selection stability', value=False,
            guide=this, position='left', text='Optional diagnostic using the upstream obs_resample training folds. Refits the entire upstream pipeline and the explored selector independently in every fold. The front shows selection frequency and the reverse shows individual decisions and failures. This can be expensive. It never changes the exported selection rule and is not an export prerequisite.'),
        ui.input_numeric(id='Seed', label='Random seed', value=2025, min=0, max=2147483647,
            guide=this, position="left", text='Controls shuffled shadows and selection-model fitting. Without an upstream design it also controls inner fold assignments; inherited designs use their own seed for those assignments. Keep seeds fixed for reproducible comparisons within the same software environment; package versions can still change recommendations.'),
        ui.input_slider(id='Workers', label='Forest worker threads', value=(1 if Module.IS_SHINYLIVE else min(4, os.cpu_count() or 1)), min=1, max=32, step=1,
            guide=this, position="left", text='Number of native LightGBM worker threads per fit. Increase cautiously when other training jobs also run in parallel. Shinylive always uses one effective worker, including exported-pipeline refits, regardless of this setting.'),
        ui.input_checkbox(id='UseWeights', label='Use any assigned observation importance weights', value=True,
            guide=this, position="left", text='Use declared observation importance weights during selection-model fitting. Balancing weights are excluded. Invalid, missing or unsuitable weights produce a status explanation rather than silently switching to unweighted fitting.'),
        ui.input_slider(id='DisplayLimit', label='Variables displayed', value=20, min=1, max=200, step=1,
            guide=this, position="left", text='Maximum number of original variables shown in the front chart, ordered by median importance. This changes only the display: all eligible variables are analyzed and remain available in the evidence table.'),
    )

    def server(input, output, session):
        busy = this.busy()
        stability_busy = this.busy()
        stability_cache = reactive.Value({})
        stability_progress = reactive.Value('')
        cache = reactive.Value({})
        progress = reactive.Value('')
        this.export_inputs('Trees','Depth','Seed','Workers','UseWeights', when=lambda: input.Recommendation() in METHODS)
        this.export_inputs('Policy','Iterations','Alpha','Percentile','TwoStep', when=lambda: input.Recommendation() == 'Boruta')
        this.export_inputs('Folds','Repeats','Cutoff', when=lambda: input.Recommendation() == 'GrootCV')

        @this.reactable(calc=True)
        def Incoming():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(source is not None)
            return source

        @this.reactable(calc=True)
        def Identity():
            return selection_identity(Incoming())

        def MethodOptions(method):
            from export_readiness import read_active_inputs
            common = {'Trees','Depth','Seed','Workers','UseWeights'}
            active = common | ({'Policy','Iterations','Alpha','Percentile','TwoStep'} if method == 'Boruta' else {'Folds','Repeats','Cutoff'})
            values = read_active_inputs(input, {'Trees':200,'Depth':5,'Seed':2025,'Workers':1,'UseWeights':True,
                'Policy':'confirmed','Iterations':100,'Alpha':.05,'Percentile':100,'TwoStep':True,
                'Folds':5,'Repeats':5,'Cutoff':1.}, active)
            return {'policy':values['Policy'],'trees':values['Trees'],'depth':values['Depth'],
                'iterations':values['Iterations'],'alpha':float(values['Alpha']),'percentile':float(values['Percentile']),
                'two_step':bool(values['TwoStep']),'folds':values['Folds'],'repeats':values['Repeats'],
                'cutoff':float(values['Cutoff']),'random_state':values['Seed'],
                'n_jobs':1 if Module.IS_SHINYLIVE else values['Workers'],'use_weights':bool(values['UseWeights'])}

        @this.reactable(calc=True)
        @this.settle(seconds=1)
        def BorutaOptions():
            return MethodOptions('Boruta')

        @this.reactable(calc=True)
        @this.settle(seconds=1)
        def GrootOptions():
            return MethodOptions('GrootCV')

        def Options(method):
            return BorutaOptions() if method == 'Boruta' else GrootOptions()

        def key(method, options):
            return (Identity(), method, tuple(sorted(options.items())))

        @this.reactable(calc=True)
        def Choice():
            choice = input.Recommendation()
            req(choice in ('None', *METHODS))
            return choice

        async def update_progress(value):
            async with reactive.lock():
                progress.set(value)
                await reactive.flush()

        @busy.track('Selecting variables…')
        @this.extended_task
        @this.record_context
        async def Calculate(source, method, options, identity):
            if Module.IS_SHINYLIVE:
                result = analyze_selection(source, method, options)
            else:
                try:
                    result = await run_selection(source, method, options, update_progress)
                except RuntimeError as error:
                    result = {'source': source, 'method': method, 'options': options, 'model': None, 'error': str(error)}
            return identity, result

        @this.reactable()
        def Start():
            source, identity = Incoming(), Identity()
            requested = Choice()
            active = input.SelectionMethod() or 'Boruta'
            results = cache.get()
            methods = [requested] if requested in METHODS else []
            methods.append(active)
            method = next((m for m in methods if key(m, Options(m)) not in results), None)
            if method is None:
                with reactive.isolate():
                    if Calculate.status() == 'running':
                        Calculate.cancel()
                return
            # Depends on cache, not progress. One native process at a time.
            Calculate.cancel()
            CalculateStability.cancel()
            progress.set('Starting selection…')
            Calculate.invoke(source.clone(), method, Options(method), identity)

        @this.reactable()
        def Completed():
            identity, result = Calculate.result()
            if identity != Identity() or result['options'] != Options(result['method']):
                return
            with reactive.isolate():
                results = {k:v for k,v in cache.get().items() if k[0] == identity}
                results[key(result['method'], result['options'])] = result
                cache.set(results)

        @this.reactable(calc=True)
        def Current():
            return cache.get().get(key(input.SelectionMethod() or 'Boruta', Options(input.SelectionMethod() or 'Boruta')))

        async def update_stability_progress(value):
            async with reactive.lock():
                stability_progress.set(value)
                await reactive.flush()

        @stability_busy.track('Checking selection stability…')
        @this.extended_task
        @this.record_context
        async def CalculateStability(source, method, options, identity):
            if Module.IS_SHINYLIVE:
                result = selection_stability(source, method, options)
            else:
                try:
                    result = await run_selection(source, method, options, update_stability_progress, diagnostic=True)
                except RuntimeError as error:
                    result = {'method':method, 'options':options, 'error':str(error)}
            return identity, result

        @this.reactable()
        def StartStability():
            if not input.Stability() or Calculate.status() == 'running':
                CalculateStability.cancel()
                return
            result = Current()
            if result is None or result['error']:
                CalculateStability.cancel()
                return
            method = input.SelectionMethod()
            options = Options(method)
            if key(method, options) in stability_cache.get():
                CalculateStability.cancel()
                return
            CalculateStability.cancel()
            stability_progress.set('Starting stability fits…')
            CalculateStability.invoke(Incoming().clone(), method, options, Identity())

        @this.reactable()
        def CompletedStability():
            identity, result = CalculateStability.result()
            if not input.Stability() or identity != Identity() or result['options'] != Options(result['method']):
                return
            with reactive.isolate():
                results = {k:v for k,v in stability_cache.get().items() if k[0] == identity}
                results[key(result['method'],result['options'])] = result
                stability_cache.set(results)

        @this.reactable(calc=True)
        def StabilityResult():
            if not input.Stability():
                return None
            return stability_cache.get().get(key(input.SelectionMethod(), Options(input.SelectionMethod())))

        @this.reactable(calc=True)
        def Selection():
            source = Incoming()
            chosen = Choice()
            if chosen == 'None':
                return {'data': source, 'result': None, 'error': '', 'pending': False}
            # Read raw settings as well as settled settings: an old cache entry
            # must stop exporting as soon as a fitting control changes.
            this.require_settled(BorutaOptions if chosen == 'Boruta' else GrootOptions)
            options = Options(chosen)
            result = cache.get().get(key(chosen, options))
            if result is None:
                return {'data': None, 'result': None, 'error': '', 'pending': True}
            try:
                data = export_selection(result)
                return {'data': data, 'result': result, 'error': '', 'pending': False}
            except ValueError as error:
                return {'data': None, 'result': result, 'error': str(error), 'pending': False}

        @this.reactable(calc=True)
        def Exported():
            data = Selection()['data']
            req(data is not None)
            return data

        @this.export_check
        def export_ready():
            Exported()

        @output
        @render.ui
        def Busy():
            return ui.TagList(busy.ui(), stability_busy.ui())

        @output
        @render.text
        def Status():
            chosen = Choice()
            if chosen != 'None':
                try:
                    selection = Selection()
                except SilentException:
                    return f'{chosen}: updating settings; downstream export paused.'
                if selection['pending']:
                    return f'{chosen}: calculating recommendation; downstream export paused. {progress.get()}'
                if selection['error']:
                    return f"{chosen}: {selection['error']} Downstream export paused; choose None or review settings."
                result = selection['result']
                suffix = ' Applied automatically; training refits the selection recipe.'
            else:
                suffix = ' None: incoming variables retained.'
                try:
                    result = Current()
                except SilentException:
                    return suffix.strip()
                if result is None:
                    return f'None: incoming variables retained. {progress.get() or "Waiting for analysis…"}'
                if result['error']:
                    return f"None: incoming variables retained. {result['error']}"
            model = result['model']
            summary = f"{result['method']}: {len(model.selected_columns_)} / {len(model.candidate_columns_)} selected; {len(model.removed_columns_)} removed."
            if not model.removed_columns_:
                summary += ' No variables removed.'
            if not model.selected_columns_:
                summary += ' No variables selected; choose None or review settings.'
            return summary + suffix

        def register_chart(method):
            @output(id=f'Chart_{method}')
            @render_widget
            def chart():
                req(this.isFront())
                result = cache.get().get(key(method, Options(method)))
                chart = stability_figure(StabilityResult(), int(input.DisplayLimit())) if input.Stability() and (result is None or not result['error']) else selection_figure(result, int(input.DisplayLimit()))
                widget = go.FigureWidget(chart)
                widget._config = {'displayModeBar':bool(this.isFullScreen()), 'displaylogo':False}
                return widget
        for method in METHODS:
            register_chart(method)

        @output
        @render.text
        def EvidenceTitle():
            return f"{input.SelectionMethod()} — decisions on original variables"

        @output
        @render.data_frame
        def Evidence():
            req(not this.isFront())
            result = Current()
            req(result is not None)
            if result['error']:
                return render.DataGrid(pd.DataFrame({'Status':[result['error']]}), height='250px')
            table = selection_evidence(result)
            stability = StabilityResult()
            if stability is not None and not stability['error']:
                table = table.merge(stability['summary'], on='Variable', how='left')
            return render.DataGrid(_display_table(table), width='100%', height='100%', filters=True)

        @output
        @render.text
        def StabilityStatus():
            if not input.Stability():
                return ''
            base = Current()
            if base is None:
                return 'Waiting for selection analysis before checking stability…'
            if base['error']:
                return 'Stability not run: ' + base['error']
            result = StabilityResult()
            if result is None:
                return stability_progress.get() or 'Waiting for stability analysis…'
            if result['error']:
                return 'Stability diagnostic: ' + result['error']
            return f"Stability: {result['successful']} / {result['planned']} training folds succeeded. Frequency is diagnostic, not a relevance probability. " + ' '.join([*result['notes'], *result['failures']])

        @output
        @render.data_frame
        def StabilityDetails():
            req(not this.isFront(), input.Stability())
            result = StabilityResult()
            req(result is not None)
            table = pd.DataFrame({'Status':[result['error']]}) if result['error'] else result['details']
            return render.DataGrid(_display_table(table), height='250px', width='100%', filters=True)

        @output
        @render.text
        def Technical():
            req(not this.isFront())
            result = Current()
            req(result is not None)
            if result['error']:
                return result['error']
            model = result['model']
            return '\n'.join([model.details_['engine'],
                f"Target: {result['target']}; task: {model.task}; fitted rows: {result['observations']}",
                f"Fits: {len(model.details_['real'])}; elapsed: {result['seconds']:.2f}s",
                f"Versions: {result['versions']}", f"Settings: {result['options']}",
                f"Effective worker threads: {model.effective_n_jobs_}",
                f"Inner resampling: {getattr(model, 'inner_resampling_', 'Not used by Boruta')}",
                'Exploratory preview (no evaluation plan).' if result['exploratory'] else 'Preview fitted on development rows only.',
                'Iteration/fold distributions are not confidence intervals. Selection is relative to this engine.',
                'Only Predictor-role variables are selected; all other roles pass through unchanged.',
                'Balancing weights are excluded; importance weights are validated without fallback.',
                'The exported recipe refits within training folds; the preview mask is not exported.'])

        session.on_ended(Calculate.cancel)
        session.on_ended(CalculateStability.cancel)
        return Exported

    this.server = server
    return this


if Module.running_directly(name=__name__):
    from proxy_data import proxy_data
    from roles import RoleMap
    rng = np.random.default_rng(2025)
    frame = pd.DataFrame({'signal':rng.normal(size=300), 'noise':rng.normal(size=300),
                          'category':pd.Categorical(rng.choice(['a', 'b', None], 300))})
    frame['outcome'] = 4 * frame.signal + rng.normal(size=300)
    this = instance()
    this._imports.set(proxy_data(frame, _roles=RoleMap.from_primitive(
        {'predictor':['signal', 'noise', 'category'], 'target':['outcome']})))
    this.run()
