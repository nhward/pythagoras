"""Search and explicitly accept a fold-evaluated pipeline configuration."""
import json
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from faicons import icon_svg as icon
from pipeline_comparison import SOURCE as COMPARISON_SOURCE
from pipeline_comparison import comparison_frame
from pipeline_search import (
    EXECUTIONS,
    METHODS,
    METRICS,
    SearchWorker,
    accepted_parameters,
    displayed,
    fresh,
    grid_coverage,
    mean_score,
    model_task,
    prepare,
    validate_checkpoint,
)
from plotly.colors import qualitative
from shiny import reactive, render, ui
from shiny.types import SilentOperationInProgressException
from shinywidgets import render_widget


def trial_frame(state):
    spec = state['specification']
    rows = []
    for t in sorted(state['trials'], key=lambda t: t['id']):
        done = t['state'] == 'complete'
        rows.append({'Trial': t['id'] + 1, 'Kind': 'Starting configuration' if t['kind'] == 'baseline' else 'Candidate', 'State': t['state'],
            'Score': displayed(mean_score(t, spec['metric']), spec['metric']) if done else None,
            'Fold SD': float(np.std([f[spec['metric']] for f in t['folds']], ddof=1)) if done and len(t['folds']) > 1 else None,
            'Folds': f"{len(t['folds'])}/{spec['folds']}", 'Seconds': t['elapsed'],
            'Metric issues': '; '.join(sorted({f'{m}: {e}' for f in t['folds'] for m, e in f.get('metric_errors', {}).items()})),
            'Method': t.get('method', t['model_id']), 'Model': t['model_id'], 'Parameters': json.dumps({k: v for k, v in t['params'].items() if k != '__method__'}), 'Error': t['error'],
            **{METRICS[spec['task']][m]: displayed(mean_score(t, m), m) if done else None for m in spec['secondary']}})
    return pd.DataFrame(rows, columns=['Trial', 'Kind', 'State', 'Score', 'Fold SD', 'Folds', 'Seconds',
                                      'Metric issues', 'Method', 'Model', 'Parameters', 'Error', *[METRICS[spec['task']][m] for m in spec['secondary']]])


def chart_snapshot(state):
    """Only completed evaluations affect the plotted scores."""
    if state is None:
        return None
    metric = state['specification']['metric']
    return deepcopy({'specification': state['specification'],
        'trials': [{'id': t['id'], 'kind': t['kind'], 'state': 'complete',
                    'params': t['params'], 'model_id': t.get('model_id', ''), 'method': t.get('method', t.get('model_id', '')), 'folds': [{metric: f[metric]} for f in t['folds']]}
                   for t in state['trials'] if t['state'] == 'complete']})


def progress_figure(state):
    metric = state['specification']['metric']
    completed = sorted((t for t in state['trials'] if t['state'] == 'complete'), key=lambda t: t['id'])
    if not completed:
        return Card.empty_figure('Waiting for the first complete evaluation; partial folds are not ranked.')
    branches = state['specification'].get('branches', {})
    def method_id(trial):
        return trial['params'].get('__method__', trial.get('model_id', trial.get('method', '')))
    methods = list(branches)
    methods.extend(key for key in dict.fromkeys(method_id(t) for t in completed) if key not in methods)
    multiple = len(methods) > 1
    figure = go.Figure()
    for index, key in enumerate(methods):
        trials = [t for t in completed if method_id(t) == key]
        if not trials:
            continue
        label = branches.get(key, {}).get('method_label') or trials[0].get('method') or key or 'Mean score'
        figure.add_trace(go.Scatter(
            x=[t['id'] + 1 for t in trials],
            y=[displayed(mean_score(t, metric), metric) for t in trials], mode='markers',
            name=label, legendgroup=key, marker={'color': qualitative.Plotly[index % len(qualitative.Plotly)], 'size': 8,
                    'symbol': ['diamond' if t['kind'] == 'baseline' else 'circle' for t in trials]},
            text=[f"{'Starting configuration' if t['kind'] == 'baseline' else 'Candidate'} · {label}" for t in trials],
            hovertemplate='Trial %{x} (%{text})<br>Mean: %{y:.6g}<extra></extra>'))
    maximum = max(t['id'] + 1 for t in completed)
    figure.update_xaxes(type='linear', range=[.5, maximum + .5], tick0=1,
                         **({'dtick': 1} if maximum <= 20 else {}))
    figure.update_xaxes(title_text='Trial')
    figure.update_yaxes(title_text=METRICS[state['specification']['task']][metric])
    figure.update_layout(title={'text': 'Mean score', 'font': {'size': 13}},
        template='plotly_white', margin={'l': 50, 'r': 30, 't': 35, 'b': 20},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#bbd6f8',
        showlegend=multiple, legend={'orientation': 'h', 'title': {'text': 'Model method'}},
        uirevision=state['specification']['fingerprint'])
    return figure


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = 'Pipeline tuning'
    this.description = 'Optimize committed pipeline and model parameters using fixed evaluation splits.'
    this.export_inputs('Metric', 'Method', 'ModelValues', 'Seed', 'Trials', 'Seconds', 'Execution', 'Workers')
    for name in ('GridCap', 'ChartView', 'SDMultiplier'):
        this.exclude_configuration_input(name)
    
    this.front = lambda: ui.TagList(
        ui.div(ui.output_ui('Best'), class_='pipeline-tune-best'),
        shinywidgets.output_widget('History', fill=True, guide=this, title='Search progress',
            text='Each point shows a completed trial’s mean primary metric across evaluation splits. '
                 'When several model methods are selected, point colors and the legend identify the method. '
                 'Diamonds identify each method’s starting configuration; circles identify tuned candidates. Partial and failed trials are not plotted.'),
    )
    
    this.back = lambda: ui.TagList(
        ui.span('Candidate evaluations · select a completed row to accept', class_='text-primary text-center d-block'),
        ui.div(ui.output_data_frame('Candidates'), class_='pipeline-tune-table html-fill-container html-fill-item'),
        ui.tags.details(ui.tags.summary('Bayesian practical-equivalence comparisons'),
            ui.output_text('ComparisonNote'),
            ui.output_data_frame('Comparisons'),
            ui.download_button('ComparisonResults', label='Download comparisons'),
            ui.a('Method and assumptions (sklearn example)', href=COMPARISON_SOURCE, target='_blank'),
            class_='pipeline-tune-details'),
        ui.tags.details(ui.tags.summary('Search space and checkpoints'),
            ui.output_text_verbatim('Specification'),
            ui.download_button('Checkpoint', label='Save checkpoint'),
            ui.download_button('Results', label='Download results'),
            ui.input_file('Restore', label='Load checkpoint JSON', accept=['.json'], multiple=False),
            class_='pipeline-tune-details'),
    )
    
    this.footer = lambda: ui.TagList(
        ui.div(
            *[ui.input_action_button(
                id=name, label=None, icon=icon(symbol, title=label, a11y='sem'),
                title=label, **{'aria-label': label},
                class_='btn rounded-pill btn-sm btn-primary',
                style='border: 0px; box-shadow: none;',
                guide = this,
                text = text
            ) for name, symbol, label, text in (
                ('Start', 'play', 'Start or resume tuning',
                 'Tune from the committed starting configuration, or resume a tuning. Trial and time budgets are cumulative.'),
                ('Pause', 'pause', 'Pause',
                 'Stop scheduling new work and finish active fits, retaining completed folds so tuning can resume.'),
                ('Cancel', 'backward', 'Cancel current tuning',
                 'Interrupt outstanding fits and pause the search. Completed folds are retained; resuming retries missing work.'),
                ('Accept', 'gavel', 'Accept selected tuning',
                 'Select a completed trial on the flip side, then accept its parameters as fixed pipeline settings. This does not perform a final fit or holdout evaluation.'),
                ('Revert', 'arrow-rotate-left', 'Revert acceptance',
                 'Clear the accepted trial and pass through the incoming pipeline settings.')
            )],
            class_='pipeline-tune-actions'
        ),
        ui.output_text('Progress'),
        # ui.output_text('Message'),
    )

    this.settings = lambda: ui.TagList(
        ui.input_select('Metric', label='Primary objective', choices={'': 'Waiting for committed model'},
            guide=this, position='left', title='Primary objective',
            text='Choose the metric that ranks completed trials. Its label states whether lower or higher is better.'),
        ui.input_select('Method', label='Search method', choices=METHODS, selected='random',
            guide=this, position='left', title='Search method',
            text='The following options are available:'
                 '<ul>'
                 '<li><strong>Random</strong> &ndash; samples unseen configurations.</li>'
                 '<li><strong>Exhaustive</strong> &ndash; evaluates all finite choices, subject to any candidate or time limit.</li>'
                 '<li><strong>Bayesian TPE</strong> &ndash; Tree-structured Parzen Estimator, learns from recorded trials to propose promising configurations.</li>'
                 '<li><strong>Bayesian GP</strong> &ndash; Gaussian-process surrogate, learns from recorded trials to propose promising configurations.</li>'
                 '</ul>'
                 'Bayesian searches do better optimization with numeric ranges. '
                 'All methods reuse the upstream resampling plan when present. Without a plan, searches use seeded '
                 'five-fold CV (target-stratified for classification), with fewer folds for small datasets. '
                 'Partition, time, spatial or grouping requirements need an explicit upstream plan.'),
        ui.input_select('Execution', label='Parallel execution', choices=EXECUTIONS, selected='auto',
            guide=this, position='left', title='Parallel execution',
            text='Auto evaluates starting-configuration folds in parallel, then parallel trials when multiple configurations remain. '
                 'Trials runs several configurations with sequential folds; Folds evaluates one configuration with parallel folds. '
                 'Sequential runs one fit at a time. Bayesian methods propose deterministic batches, so the batch width can change the search path.'),
        ui.input_numeric('Workers', label='Maximum simultaneous fits (0 = automatic)', value=0, min=0, max=64, step=1,
            guide=this, position='left', title='Workers',
            text='Zero chooses half the available CPUs, rounded down, with at least one worker. '
                 'Each fit uses one estimator/native-library thread. Changing this setting starts a different search identity.'),
        ui.input_numeric('Trials', label='Maximum candidates excluding starting configurations (0 = unlimited)', value=30, min=0, max=10000, step=1,
            guide=this, position='left', title='Candidate budget',
            text='Maximum candidate trials across all committed methods, excluding one starting-configuration evaluation per method. Zero removes the candidate limit for every search method; the search still stops when all finite configurations have been evaluated or the time limit is reached. Budgets are cumulative across resumes.'),
        ui.input_numeric('EquivalenceTolerance', label='Practical-equivalence tolerance (metric units)', value=None, min=0,
            guide=this, position='left', title='Practical-equivalence tolerance',
            text='Required for Bayesian comparisons. Differences within plus or minus this value are practically equivalent. '
                 'Use absolute primary-metric units: 0.01 accuracy is one percentage point; MAE uses response units. '
                 'Choose this using application needs, ideally before examining results. This affects the comparison table only; it does not alter the search or chart.'),
        ui.input_numeric('ComparisonLimit', label='Top trials to compare', value=5, min=2, max=20, step=1,
            guide=this, position='left', title='Comparison shortlist',
            text='Compare every pair among this many top completed trials. Starting configurations participate if they are in the shortlist. '
                 'Incomplete or failed trials are excluded. This limits quadratic table growth; it does not change search results.'),
        ui.input_numeric('DecisionProbability', label='Comparison decision probability', value=0.95, min=0.501, max=0.999, step=0.01,
            guide=this, position='left', title='Comparison decision probability',
            text='Label a pair better, worse or practically equivalent only when its posterior probability reaches this threshold. '
                 'Otherwise report Inconclusive. Credible intervals always have 95% posterior coverage. Acceptance remains manual.'),
        ui.input_numeric('Seconds', label='Total time budget in seconds (0 = unlimited)', value=0, min=0,
            guide=this, position='left', title='Time budget',
            text='Cumulative time budget across resumes. Zero means unlimited. Time is checked between folds; Cancel current fit can interrupt a long fit.'),
        ui.input_numeric('Seed', label='Random seed', value=1729, min=0, max=2147483647, step=1,
            guide=this, position='left', title='Random seed',
            text='Allows search method and modeling method to be reproducible. Changing it creates a different tuning identity.'),
        ui.input_text_area('ModelValues', label='Additional model choices (JSON)', value='{}', rows=5,
            placeholder='{"data_models__estimator__alpha": [0.1, 1, 10]}',
            guide=this, position='left', title='Additional model choices',
            text='Add full model parameter paths with finite lists of candidate values. Existing committed choices must be edited in Tuning parameters.'),
    )

    def server(input, output, session):
        restored = this.restored_configuration_field('search', {})
        state = reactive.Value(restored.get('checkpoint'))
        chart_state = reactive.Value(None)
        accepted = reactive.Value(restored.get('accepted'))
        message = reactive.Value('')
        worker = SearchWorker()
        session.on_ended(worker.close)
        this.register_configuration_state_provider('search', lambda: {
            'checkpoint': state(), 'accepted': accepted()})

        @this.reactable(calc=True)
        def Incoming():
            try:
                return this.input_data()
            except SilentOperationInProgressException:
                return None

        @this.reactable()
        def Metrics():
            source = Incoming()
            if source is None: return
            try: choices = METRICS[model_task(source)]
            except ValueError: return
            with reactive.isolate(): current = input.Metric()
            ui.update_select('Metric', choices=choices, selected=current if current in choices else next(iter(choices)))

        @this.reactable(calc=True)
        def Problem():
            try:
                if Incoming() is None: raise ValueError('Waiting for upstream data.')
                if sys.platform == 'emscripten': raise ValueError('Pipeline tuning requires the native Python server; isolated fitting is unavailable in this browser runtime.')
                seed = input.Seed()
                if seed is None or int(seed) != seed: raise ValueError('Choose an integer search seed.')
                workers = input.Workers()
                if workers is None or int(workers) != workers:
                    raise ValueError('Choose an integer worker limit.')
                return prepare(Incoming(), model_values=json.loads(input.ModelValues() or '{}'),
                    metric=input.Metric() or None,
                    method=input.Method(), seed=int(seed), execution=input.Execution() or 'auto', workers=int(workers)), ''
            except (ValueError, TypeError, OverflowError) as exc:
                return None, str(exc)

        def compatible():
            problem, error = Problem()
            if problem is None: raise ValueError(error)
            current = state()
            if current is None: raise ValueError('Start a search first.')
            return problem, validate_checkpoint(problem, current)

        @reactive.effect
        def UpdateChart():
            snapshot = chart_snapshot(state())
            with reactive.isolate(): previous = chart_state()
            if snapshot != previous:
                chart_state.set(snapshot)

        @reactive.effect
        def Poll():
            active = this._active()
            state()  # Play also wakes this poller when it was idle.
            if not active: worker.pause()
            latest = worker.poll()
            if latest is not None: state.set(latest)
            if worker.running:
                reactive.invalidate_later(.3)
            elif worker.process is not None:
                with reactive.isolate(): previous = state()
                if previous and previous['status'] == 'running':
                    state.set(dict(previous, status='paused', error='Worker exited; completed folds were preserved.'))

        @this.reactable(triggers=[input.Start])
        def Start():
            try:
                problem, error = Problem()
                if problem is None: raise ValueError(error)
                if worker.running: raise ValueError('Pause the current search before starting another.')
                current = state()
                if (current is not None and current['status'] != 'exhausted'
                        and current['specification']['fingerprint'] == problem.specification['fingerprint']):
                    checkpoint = validate_checkpoint(problem, current)
                    worker.start(problem, checkpoint, *_budgets())
                    state.set(dict(checkpoint, status='running'))
                    message.set('Fitting resumed; budgets are cumulative.')
                    return
                initial = fresh(problem)
                worker.start(problem, initial, *_budgets())
                state.set(dict(initial, status='running')); accepted.set(None)
                message.set('Fitting started. Each method’s committed starting configuration is evaluated first.')
            except (ValueError, TypeError, RuntimeError) as exc: message.set(str(exc))

        def _budgets():
            trials, seconds = input.Trials(), input.Seconds()
            if trials is None or int(trials) != trials or not 0 <= trials <= 10000:
                raise ValueError('Choose a candidate budget between 0 and 10,000; zero means unlimited.')
            if seconds is None or not np.isfinite(seconds) or seconds < 0:
                raise ValueError('Choose a nonnegative total time budget.')
            return int(trials), float(seconds)

        @this.reactable(triggers=[input.Pause])
        def Pause():
            worker.pause(); message.set('Pause requested; the current fold will finish and be checkpointed.')

        @this.reactable(triggers=[input.Cancel])
        def Cancel():
            latest = worker.poll()
            if latest is not None: state.set(latest)
            elapsed = worker.elapsed if worker.running else 0.
            worker.close()
            if state() is not None: state.set(dict(state(), status='paused', elapsed=max(state()['elapsed'], elapsed)))
            message.set('Current fit cancelled. Completed folds are retained; Resume retries the interrupted fold.')

        @this.reactable(triggers=[input.Restore])
        def Restore():
            files = input.Restore()
            if not files: return
            try:
                if worker.running: raise ValueError('Pause the search before loading a checkpoint.')
                if files[0]['size'] > 20_000_000: raise ValueError('Checkpoint must be smaller than 20 MB.')
                problem, error = Problem()
                if problem is None: raise ValueError(error)
                with open(files[0]['datapath']) as stream: loaded = json.load(stream)
                state.set(validate_checkpoint(problem, loaded)); accepted.set(None)
                message.set('Compatible checkpoint loaded. Resume to continue.')
            except (ValueError, TypeError, KeyError, OSError) as exc: message.set(str(exc))

        @this.reactable(triggers=[input.Accept])
        def Accept():
            try:
                problem, checkpoint = compatible()
                selected = Candidates.data_view(selected=True)
                if selected.empty: raise ValueError('Select a completed candidate on the flip side first.')
                trial_id = int(selected.iloc[0]['Trial']) - 1
                accepted_parameters(problem, checkpoint, trial_id)
                accepted.set({'fingerprint': problem.specification['fingerprint'], 'trial': trial_id})
                message.set(f"Trial {trial_id + 1} accepted as fixed parameters. Method: {checkpoint['trials'][trial_id].get('method', checkpoint['trials'][trial_id]['model_id'])}. No final fit or holdout evaluation was performed.")
            except (ValueError, TypeError) as exc: message.set(str(exc))

        @this.reactable(triggers=[input.Revert])
        def Revert():
            accepted.set(None); message.set('Acceptance reverted; incoming pipeline settings pass through.')

        @output
        @render.text
        def Message(): return message()

        @output
        @render.text
        def Progress():
            problem, error = Problem()
            if problem is None: return error
            current = state()
            evaluation = problem.specification['evaluation']
            execution = problem.specification['execution']
            repeat_label = (f" · {evaluation['repeats']} repeats × {evaluation['folds_per_repeat']} folds"
                            if evaluation.get('repeats', 1) > 1 else '')
            split_label = (f"{evaluation['origin'].title()} {evaluation['strategy']} resampling{repeat_label}"
                           f" · {EXECUTIONS[execution['mode']]} · {execution['workers']} workers")
            if current is None: return f"Ready · {len(problem.branches) or 1} starting-configuration evaluations · {problem.specification['combinations']:,} configurations · {problem.specification['folds']} folds · {split_label}"
            if current['specification']['fingerprint'] != problem.specification['fingerprint']:
                return 'Search is stale: inputs changed. Restore the original settings to resume, or start a new search.'
            completed = [t for t in current['trials'] if t['state'] == 'complete']
            best = max(completed, key=lambda t: mean_score(t, problem.specification['metric']), default=None)
            detail = f" · Best so far: trial {best['id'] + 1} ({displayed(mean_score(best, problem.specification['metric']), problem.specification['metric']):.5g})" if best else ''
            active = [t for t in current['trials'] if t['state'] == 'running']
            fold = (f" · {len(active)} active trials: {sum(len(t['folds']) for t in active)}/"
                    f"{len(active) * problem.specification['folds']} splits completed") if active else ''
            coverage_text = ''
            if current['specification']['method'] == 'grid':
                coverage = grid_coverage(current)
                label = 'Exhaustive search' if coverage['visited'] == coverage['total'] else 'Partial exhaustive search'
                coverage_text = (f" · {label}: {coverage['complete']} of {coverage['total']} grid configurations completed"
                                 f"; {coverage['failed']} failed; starting configurations evaluated separately")
            status_label = {'running': 'Running', 'exhausted': 'Finished', 'budget reached': 'Halted', 'paused': 'Halted'}.get(current['status'], current['status'])
            return f"{status_label} · {len(completed)} complete · {sum(t['state'] == 'failed' for t in current['trials'])} failed · {current['elapsed']:.1f}s{detail}{fold}" + f' · {split_label}' + coverage_text + (f" · {current['error']}" if current.get('error') else '')

        @output
        @render.ui
        def Best():
            current = chart_state()
            if current is None: return None
            completed = [t for t in current['trials'] if t['state'] == 'complete']
            if not completed: return None
            spec = current['specification']
            best = max(completed, key=lambda t: mean_score(t, spec['metric']))
            problem, _ = Problem()
            stale = problem is None or spec['fingerprint'] != problem.specification['fingerprint']
            baseline = next((t for t in completed if t['kind'] == 'baseline' and t.get('model_id') == best.get('model_id')), None)
            improvement = (f" · Improvement over starting configuration: {mean_score(best, spec['metric']) - mean_score(baseline, spec['metric']):.6g}"
                           if baseline else '')
            values = [ui.span(ui.code(path), ' = ', ui.code(json.dumps(value)), '; ')
                      for path, value in sorted(best['params'].items()) if path != '__method__']
            return ui.TagList(
                ui.strong(f"{'Previous search (stale): best' if stale else 'Best parameters found so far'} · Trial {best['id'] + 1} · {best.get('method', best.get('model_id', ''))}"),
                ui.div(f"{METRICS[spec['task']][spec['metric']]}: {displayed(mean_score(best, spec['metric']), spec['metric']):.6g}{improvement}"),
                ui.div(*values) if values else ui.div('No parameters enabled; the committed configuration was evaluated.'),
            )

        @output
        @render_widget
        def History():
            current = chart_state()
            figure = progress_figure(current) if current else Card.empty_figure('Start a search to compare completed configurations.')
            figure.update_layout(modebar={'orientation': 'v'},
                showlegend=bool(figure.layout.showlegend) and bool(this.isFullScreen()))
            widget = go.FigureWidget(figure)
            widget._config = {'displayModeBar': bool(this.isFullScreen()), 'displaylogo': False, 'responsive': True}
            return widget

        @output
        @render.data_frame
        def Candidates():
            current = state()
            return render.DataGrid(trial_frame(current).round(3) if current else pd.DataFrame(columns=['Trial', 'State', 'Score']),
                                   selection_mode='row', height='100%', filters=False)

        @reactive.calc
        def Comparison():
            current = state()
            problem, _ = Problem()
            if current is not None and (problem is None or current['specification']['fingerprint'] != problem.specification['fingerprint']):
                return pd.DataFrame(), 'Search results are stale; restore matching settings or run a new search.'
            limit = input.ComparisonLimit()
            if limit is not None and float(limit).is_integer():
                limit = int(limit)
            probability = input.DecisionProbability()
            return comparison_frame(current, input.EquivalenceTolerance(), limit=limit,
                                    probability=0.95 if probability is None else probability)

        @output
        @render.text
        def ComparisonNote():
            return Comparison()[1]

        @output
        @render.data_frame
        def Comparisons():
            return render.DataGrid(Comparison()[0].round(3), height='300px', filters=False)

        @output
        @render.download(filename='pipeline-comparisons.csv')
        def ComparisonResults():
            yield Comparison()[0].to_csv(index=False)

        @output
        @render.text
        def Specification():
            problem, error = Problem()
            return json.dumps(problem.specification, indent=2) if problem else error

        @output
        @render.download(filename='pipeline-search.json')
        def Checkpoint():
            if state() is not None: yield json.dumps(state(), allow_nan=False, indent=2)

        @output
        @render.download(filename='pipeline-search.csv')
        def Results():
            if state() is not None: yield trial_frame(state()).to_csv(index=False)

        @this.reactable(calc=True)
        def Exported():
            source = Incoming()
            if source is None: return None
            selection = accepted()
            if selection is not None:
                try:
                    problem, error = Problem()
                    if problem is None: raise ValueError(error)
                    # Live progress must not republish an accepted pipeline or
                    # repeatedly invalidate downstream cards.
                    with reactive.isolate(): checkpoint = validate_checkpoint(problem, state())
                    if selection['fingerprint'] != problem.specification['fingerprint']:
                        raise ValueError('Acceptance is stale.')
                    parameters = accepted_parameters(problem, checkpoint, selection['trial'])
                    selected = checkpoint['trials'][selection['trial']]
                    return source.with_tuned_parameters(parameters, fingerprint=selection['fingerprint'],
                        trial=selection['trial'], metric=problem.specification['metric'],
                        candidate_id=selected['params'].get('__method__'),
                        score=mean_score(selected, problem.specification['metric']))
                except (ValueError, TypeError, KeyError): pass
            return source.with_inactive_step(stage='Learning', card='pipeline_tune', operation='Accept tuned pipeline parameters')

        return Exported

    this.server = server
    return this
