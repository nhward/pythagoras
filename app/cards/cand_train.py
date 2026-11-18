"""Search and publish evidence for fold-evaluated pipeline configurations."""
import json
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import prediction_evidence as evidence_store
import shinywidgets
from card import Card
from faicons import icon_svg as icon
from model_review import archived_evidence, completed_evidence
from pipeline_search import (
    EXECUTIONS,
    METHODS,
    METRICS,
    SearchWorker,
    displayed,
    fresh,
    grid_coverage,
    mean_score,
    model_task,
    prepare,
    validate_checkpoint,
)
from plotly.colors import qualitative
from reactive_timer import invalidate_later
from search_publication import CompletedTrialPublication
from shiny import reactive, render, req, ui
from shiny.types import SilentOperationInProgressException
from shinywidgets import render_widget
from widget_lifecycle import WidgetSlot, update_figure

#TODO: Reviw quarto

def best_parameter_label(path, trial):
    """Format a tuning path for the best-result summary only."""
    root, *parts = path.split('__')
    if root == 'cand_methods':
        title = trial.get('method', trial.get('model_id', 'Model'))
        for suffix in (' Regressor', ' Classifier', ' Regression', ' Classification'):
            title = title.removesuffix(suffix)
        if parts and parts[0] == 'estimator':
            parts = parts[1:]
    else:
        title = root.replace('_', ' ').capitalize()
    return f"{title} ({' › '.join(parts)})" if parts else title


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
    source_spec = state['specification']
    metric = source_spec['metric']
    chart_spec = {k: source_spec[k] for k in ('metric', 'task', 'fingerprint')}
    if 'combinations' in source_spec:
        chart_spec['combinations'] = source_spec['combinations']
    if source_spec.get('branches'):
        chart_spec['branches'] = {key: {'method_label': value['method_label']} for key, value in source_spec['branches'].items()}
    return deepcopy({'specification': chart_spec,
        'trials': [{'id': t['id'], 'kind': t['kind'], 'state': 'complete',
                    'params': t['params'], 'model_id': t.get('model_id', ''), 'method': t.get('method', t.get('model_id', '')), 'folds': [{metric: f[metric]} for f in t['folds']]}
                   for t in state['trials'] if t['state'] == 'complete']})


def history_axis_ranges(state, previous_y=None):
    spec = state['specification']
    trials = sorted((t for t in state['trials'] if t['state'] == 'complete'), key=lambda t: t['id'])
    maximum = max((t['id'] + 1 for t in trials), default=1)
    planned = spec.get('combinations', 0) + max(1, len(spec.get('branches', {})))
    x_range = [.5, max(maximum, planned) + .5]
    values = [displayed(mean_score(t, spec['metric']), spec['metric']) for t in trials]
    if not values:
        return x_range, previous_y
    lo, hi = previous_y or (min(0., 2 * values[0]), max(0., 2 * values[0]))
    if hi == lo:
        lo, hi = 0., 1.
    while min(values) < lo:
        lo -= hi - lo
    while max(values) > hi:
        hi += hi - lo
    return x_range, [lo, hi]


def progress_figure(state, *, axis_ranges=None):
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
        figure.add_trace(go.Bar(
            x=[t['id'] + 1 for t in trials],
            y=[displayed(mean_score(t, metric), metric) for t in trials], width=.8,
            name=label, legendgroup=key, marker={'color': qualitative.Plotly[index % len(qualitative.Plotly)],
                    'line': {'width': 0}, 'opacity': 1,
                    'pattern': {'shape': ['/' if t['kind'] == 'baseline' else '' for t in trials]}},
            text=[f"{'Starting configuration' if t['kind'] == 'baseline' else 'Candidate'} · {label}" for t in trials],
            hovertemplate='Trial %{x} (%{text})<br>Mean: %{y:.6g}<extra></extra>'))
    x_range, y_range = axis_ranges or history_axis_ranges(state)
    maximum = x_range[1] - .5
    figure.update_xaxes(
        title_text='Trial',
        type='linear', 
        range=x_range, 
        tick0=1,
        **({'dtick': 1} if maximum <= 20 else {})
    )
    figure.update_yaxes(title_text=METRICS[state['specification']['task']][metric], range=y_range)
    figure.update_layout(
        title={
            'text': 'Trial score',
            'font': {'size':17},
            'x': 0.5, 
            'xanchor': "center",
            'y': 1, 
            'yanchor': "top",
        },
        template='plotly_white', margin={'l': 50, 'r': 30, 't': 15, 'b': 5},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#bbd6f8',
        barmode='overlay', showlegend=multiple, legend={'orientation': 'h', 'title': {'text': 'Model method'}},
        uirevision=state['specification']['fingerprint']
    )
    return figure


def validate_legacy_choices(legacy_choices, tuning_spec):
    """Require old local choices to be explicitly committed upstream."""
    try:
        choices = json.loads(legacy_choices or '{}') if isinstance(legacy_choices, str) else legacy_choices
    except (ValueError, TypeError):
        choices = None
    grid = (tuning_spec or {}).get('grid', {})
    if not isinstance(choices, dict) or any(grid.get(path) != values for path, values in choices.items()):
        raise ValueError('This saved card contains Additional model choices. Commit those choices in Tuning parameters before training: ' + str(legacy_choices))


def instance():
    this = Card(file=__file__, mutable=True)
    this.register_configuration_state_field('search')
    this.register_configuration_state_field('legacy_model_values')
    this.exclude_configuration_input('Secondary')
    this.long_name = 'Pipeline training'
    this.description = 'Train pipelines across candidate methods and hyper-parameters.'
    this.export_inputs('Metric', 'Method', 'Seed', 'Trials', 'Seconds', 'Execution', 'Workers')
    for name in ('GridCap', 'ChartView', 'SDMultiplier', 'EquivalenceTolerance', 'ComparisonLimit', 'DecisionProbability'):
        this.exclude_configuration_input(name)
    
    this.front = lambda: ui.TagList(
        ui.div(ui.output_ui('Best'), class_='pipeline-tune-best',
               tabindex='0', role='region', **{'aria-label': 'Best tuning result'}),
        shinywidgets.output_widget('History', fill=True, guide=this, title='Search progress',
            text='Each bar shows a completed trial’s mean primary metric across evaluation splits. '
                 'When several model methods are selected, bar colors and the legend identify the method. '
                 'Hatched bars identify each method’s starting configuration; solid bars identify tuned candidates. Partial and failed trials are not plotted.'),
    )
    
    this.back = lambda: ui.TagList(
        ui.span('Candidate evaluations', class_='text-primary text-center d-block'),
        ui.div(ui.output_data_frame('Candidates'), class_='pipeline-tune-table html-fill-container html-fill-item'),

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
                 'Tune from the committed starting configuration, or resume a tuning. Trial and time budgets are cumulative for non-exhaustive searches.'),
                ('Pause', 'pause', 'Pause',
                 'Stop scheduling new work and finish active fits, retaining completed folds so tuning can resume.'),
                ('Cancel', 'backward-fast', 'Cancel current tuning',
                 'Stop fitting and discard all trials and progress. Play starts a new search; use Pause to retain results.')
            )],
            class_='pipeline-tune-actions'
        ),
        ui.div(ui.output_text('Progress'), class_='pipeline-tune-progress',
               tabindex='0', role='region', **{'aria-label': 'Tuning progress'}),
        # ui.output_text('Message'),
    )

    this.settings = lambda: ui.TagList(
        ui.input_select('Metric', label='Primary objective', choices={'': 'Waiting for committed model'},
            guide=this, position='left', title='Primary objective',
            text='Choose the metric that ranks completed trials. Its label states whether lower or higher is better.'),
        ui.input_radio_buttons('Method', label='Search method', choices=METHODS, selected='random',
            guide=this, position='left', title='Search method',
            text='The following options are available:'
                 '<ul>'
                 '<li><strong>Random</strong> &ndash; samples unseen configurations.</li>'
                 '<li><strong>Exhaustive</strong> &ndash; evaluates all finite choices unless manually paused; candidate and time budgets are ignored.</li>'
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
            text='Ignored for Exhaustive. Maximum candidate trials across all committed methods, excluding one starting-configuration evaluation per method. Zero removes the candidate limit; the search still stops when all finite configurations have been evaluated or the time limit is reached. Budgets are cumulative across resumes.'),
        ui.input_numeric('Seconds', label='Total time budget in seconds (0 = unlimited)', value=0, min=0,
            guide=this, position='left', title='Time budget',
            text='Ignored for Exhaustive. Cumulative time budget across resumes. Zero means unlimited. Time is checked between folds; Cancel current fit can interrupt a long fit.'),
        ui.input_numeric('RefreshSeconds', label='Result refresh interval (seconds)', value=5, min=0, step=1,
            guide=this, position='left', title='Result refresh interval',
            text='Refresh the completed-trial chart and downstream model review together at most once per interval while tuning. '
                 'Zero publishes every completed update. Progress and worker communication stay responsive. '
                 'Final results are published immediately. This setting does not change the search identity.'),
        ui.input_numeric('Seed', label='Random seed', value=1729, min=0, max=2147483647, step=1,
            guide=this, position='left', title='Random seed',
            text='Allows search method and modeling method to be reproducible. Changing it creates a different tuning identity.'),
    )

    def server(input, output, session):
        legacy_choices = this.consume_restored_configuration_field('legacy_model_values', None)
        if legacy_choices is None:
            legacy_choices = this.restored_configuration_input('ModelValues', '{}', warn_if_missing=False)
        this.exclude_configuration_input('ModelValues')
        this.register_configuration_state_provider('legacy_model_values', lambda: legacy_choices)
        history_widget = WidgetSlot(session)
        history_instance = reactive.Value(None)
        history_fingerprint = None
        history_y_range = None
        restored = this.consume_restored_configuration_field('search', {})
        restored = restored if isinstance(restored, dict) else {}
        restore_error = ''
        try:
            initial_checkpoint = evidence_store.restore_checkpoint(restored.pop('checkpoint', None))
            if initial_checkpoint is not None and initial_checkpoint.get('legacy_evidence_removed'):
                restore_error = 'Old row-level predictions were discarded; start a new search for compact diagnostics. Score history is preserved.'
        except (ValueError, OSError) as exc:
            initial_checkpoint = None
            restore_error = str(exc)
        state = reactive.Value(initial_checkpoint)
        table_state = reactive.Value(initial_checkpoint)
        table_published_at = 0.
        evidence_source = reactive.Value(restored.get('source_signature'))
        chart_state = reactive.Value(None)
        published_state = reactive.Value(None)
        publication = CompletedTrialPublication()
        message = reactive.Value(restore_error)
        failure_notice = None

        @reactive.effect
        def NotifyFailures():
            nonlocal failure_notice
            current = state()
            failures = [] if current is None else [t for t in current['trials'] if t['state'] == 'failed'][:10]
            notice = None if not failures else (
                current['specification']['fingerprint'],
                tuple((t['id'], t['error']) for t in failures))
            if notice == failure_notice:
                return
            failure_notice = notice
            notification_id = this.ns('FitFailures')
            if notice is None:
                ui.notification_remove(notification_id)
                return
            ui.notification_show(ui.TagList(
                ui.strong('Training failures (first 10 maximum)'),
                ui.p('The complete failure list is available in the flip-side results table.'),
                ui.tags.ol(*[ui.tags.li(
                    ui.strong(f"Trial {t['id'] + 1} · {t.get('method', t['model_id'])}"),
                    ui.tags.pre(json.dumps({k: v for k, v in t['params'].items() if k != '__method__'}),
                                style='white-space: pre-wrap; overflow-wrap: anywhere;'),
                    ui.div(t['error'] or 'Fit failed without an error message.', style='overflow-wrap: anywhere;'))
                    for t in failures], style='max-height: 20rem; overflow-y: auto; padding-left: 1.5rem;')),
                id=notification_id, type='error', duration=None, close_button=True)

        worker = SearchWorker()
        
        def cleanup():
            worker.close()
            with reactive.isolate(): current = state()
            evidence_store.remove(evidence_store.references(current))
        
        session.on_ended(cleanup)
        
        this.register_configuration_state_provider('search', lambda: {
            'checkpoint': evidence_store.portable_checkpoint(state()),
            'source_signature': evidence_source()})

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
            nonlocal legacy_choices
            try:
                if Incoming() is None: raise ValueError('Waiting for upstream data.')
                if sys.platform == 'emscripten': raise ValueError('Pipeline tuning requires the native Python server; isolated fitting is unavailable in this browser runtime.')
                seed = input.Seed()
                if seed is None or int(seed) != seed: raise ValueError('Choose an integer search seed.')
                workers = input.Workers()
                if workers is None or int(workers) != workers:
                    raise ValueError('Choose an integer worker limit.')
                validate_legacy_choices(legacy_choices, Incoming().tuning_spec)
                legacy_choices = '{}'
                return prepare(Incoming(),
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

        def refresh_interval():
            interval = input.RefreshSeconds()
            if interval is None or not np.isfinite(interval) or interval < 0:
                interval = 5
            return interval

        @reactive.effect
        def PublishTable():
            nonlocal table_published_at
            current = state()
            interval = refresh_interval()
            with reactive.isolate(): previous = table_state()
            if current is previous:
                return
            now = time.monotonic()
            immediate = (current is None or previous is None or current['status'] != 'running'
                         or current['specification']['fingerprint'] != previous['specification']['fingerprint']
                         or len(current['trials']) < len(previous['trials']))
            remaining = interval - (now - table_published_at)
            if not immediate and remaining > 0:
                invalidate_later(remaining)
                return
            table_published_at = now
            table_state.set(current)

        @reactive.effect
        def UpdateChart():
            interval = refresh_interval()
            changed, retry = publication.update(state(), interval, time.monotonic())
            if retry is not None:
                invalidate_later(retry)
            if not changed:
                return
            published_state.set(publication.snapshot)
            snapshot = chart_snapshot(publication.snapshot)
            with reactive.isolate(): previous = chart_state()
            if snapshot != previous:
                chart_state.set(snapshot)

        @reactive.calc
        def ReviewResults():
            current = published_state()
            problem, _ = Problem()
            if current is None or not current['trials']:
                return None
            if problem is not None and current['specification']['fingerprint'] == problem.specification['fingerprint']:
                return completed_evidence(problem, current)
            source = Incoming()
            if evidence_source() is not None and source is not None and evidence_source() == source._tuning_evidence_signature():
                return archived_evidence(source, current)
            return None


        @reactive.effect
        def Poll():
            active = this._active()
            state()  # Play also wakes this poller when it was idle.
            if not active: 
                worker.pause()
            latest = worker.poll()
            if latest is not None: 
                state.set(latest)
            if worker.running:
                invalidate_later(.3)
            elif worker.process is not None:
                with reactive.isolate(): previous = state()
                if previous and previous['status'] == 'running':
                    state.set(dict(previous, status='paused', error='Worker exited; completed folds were preserved.'))

        @this.reactable(triggers=[input.Start])
        def Start():
            try:
                problem, error = Problem()
                if problem is None: 
                    raise ValueError(error)
                if worker.running: raise ValueError('Pause the current search before starting another.')
                current = state()
                if (current is not None and current['status'] != 'exhausted'
                        and current['specification']['fingerprint'] == problem.specification['fingerprint']):
                    checkpoint = validate_checkpoint(problem, current)
                    worker.start(problem, checkpoint, *_budgets())
                    state.set(dict(checkpoint, status='running'))
                    message.set('Fitting resumed; Exhaustive ignores candidate and time budgets.' if problem.specification['method'] == 'grid' else 'Fitting resumed; budgets are cumulative.')
                    this.log.info("Pipeline fitting resumed")
                    return
                if current is not None:
                    message.set('Cannot resume this search; history retained. Cancel to clear it before starting again.')
                    return
                start_fresh(problem)
            except (ValueError, TypeError, RuntimeError) as exc: 
                e = str(exc)
                this.log.error(e)
                message.set(e)

        def start_fresh(problem):
            initial = fresh(problem)
            budgets = _budgets()
            worker.close()
            worker.start(problem, initial, *budgets)
            evidence_store.remove(evidence_store.references(state()))
            evidence_source.set(Incoming()._tuning_evidence_signature())
            state.set(dict(initial, status='running'))
            message.set('Fitting started. Each method’s committed starting configuration is evaluated first.')
            this.log.info("Pipeline fitting started")

        def _budgets():
            if input.Method() == 'grid':
                return 0, 0.
            trials, seconds = input.Trials(), input.Seconds()
            if trials is None or int(trials) != trials or not 0 <= trials <= 10000:
                raise ValueError('Choose a candidate budget between 0 and 10,000; zero means unlimited.')
            if seconds is None or not np.isfinite(seconds) or seconds < 0:
                raise ValueError('Choose a nonnegative total time budget.')
            return int(trials), float(seconds)

        @this.reactable(triggers=[input.Pause])
        def Pause():
            worker.pause(); message.set('Pause requested; the current fold will finish and be checkpointed.')
            this.log.info("Fitting paused")

        @this.reactable(triggers=[input.Cancel])
        def Cancel():
            latest = worker.poll()
            references = evidence_store.references(state()) | evidence_store.references(latest)
            worker.close()
            evidence_store.remove(references)
            evidence_source.set(None)
            state.set(None)
            message.set('Tuning cancelled and history cleared. Play starts a new search.')
            this.log.info("Fitting cancelled")

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
            split_label = f"{evaluation['origin'].title()} {evaluation['strategy']} resampling{repeat_label} · {execution['workers']} workers"
            if current is None: return f"Ready · {len(problem.branches) or 1} starting-configuration evaluations · {problem.specification['combinations']:,} configurations · {problem.specification['folds']} folds · {split_label}"
            if current['specification']['fingerprint'] != problem.specification['fingerprint']:
                return 'Search is stale: inputs changed. Restore the original settings to resume, or Cancel to clear history and start again.'
            completed = [t for t in current['trials'] if t['state'] == 'complete']
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
            return f"{status_label} · {len(completed)} complete · {sum(t['state'] == 'failed' for t in current['trials'])} failed · {current['elapsed']:.1f}s{fold}" + f' · {split_label}' + coverage_text + (f" · {current['error']}" if current.get('error') else '')

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
            values = [ui.span(best_parameter_label(path, best), ' = ', ui.code(json.dumps(value)), '; ')
                      for path, value in sorted(best['params'].items()) if path != '__method__']
            return ui.TagList(
                ui.strong(f"{'Previous search (stale): best' if stale else 'Best parameters found so far'} · Trial {best['id'] + 1} · {best.get('method', best.get('model_id', ''))}"),
                ui.div(f"{METRICS[spec['task']][spec['metric']]}: {displayed(mean_score(best, spec['metric']), spec['metric']):.4g}"),
                ui.div(*values) if values else ui.div('No parameters enabled; the committed configuration was evaluated.'),
            )

        @output
        @render_widget
        def History():
            widget = history_widget.replace(go.FigureWidget())
            widget._config = {'displayModeBar': False, 'displaylogo': False, 'responsive': True}
            history_instance.set(widget)
            return widget

        @reactive.effect
        async def UpdateHistory():
            nonlocal history_fingerprint, history_y_range
            widget = history_instance()
            req(widget is not None)
            current = chart_state()
            fingerprint = current['specification']['fingerprint'] if current else None
            if fingerprint != history_fingerprint or not current or not current['trials']:
                history_y_range = None
            history_fingerprint = fingerprint
            if current:
                x_range, history_y_range = history_axis_ranges(current, history_y_range)
                figure = progress_figure(current, axis_ranges=(x_range, history_y_range))
            else:
                figure = Card.empty_figure('Start a search to compare completed configurations.')
            figure.update_layout(modebar={'orientation': 'v'},
                showlegend=bool(figure.layout.showlegend) and bool(this.isFullScreen()))
            config = {'displayModeBar': bool(this.isFullScreen()), 'displaylogo': False, 'responsive': True}
            if update_figure(widget, figure, config, preserve_traces=True):
                await session.send_custom_message('update_plotly_config', {'id': this.ns('History'), 'config': config})

        @output
        @render.data_frame
        def Candidates():
            current = table_state()
            return render.DataGrid(
                trial_frame(current).round(3) if current else pd.DataFrame(columns=['Trial', 'State', 'Score']),
                selection_mode='row', 
                height='100%', 
                filters=False
            )

        @this.reactable(calc=True)
        def Exported():
            source = Incoming()
            if source is None: return None
            results = ReviewResults()
            if results is not None:
                source = source.with_tuning_results(results)
            return source

        return Exported

    this.server = server
    return this
