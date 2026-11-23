"""Evaluate ensembles of weighted method winners without final fitting."""
import asyncio
import threading

import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from code_recording import recordable
from ensemble import (
    EnsembleDisplay,
    EnsembleEvaluation,
    evaluate_ensemble,
    evaluation_context_signature,
    evaluation_signature,
    portable_ensemble_display,
    representatives,
    restore_ensemble_display,
)
from ensemble_diagnostics import performance_figure
from fit_service import session_fit_service
from module import Module
from pipeline_search import METRICS
from shiny import reactive, render, req, ui
from shiny.types import SilentException
from shinywidgets import render_widget
from widget_lifecycle import WidgetSlot, update_figure

STYLES = {'regression': {'mean': 'Equal mean', 'median': 'Median'},
          'classification': {'soft': 'Soft voting (mean probabilities)', 'hard': 'Hard majority voting'}}


@recordable
def _figure(result, metric, full_screen=False, class_index=0, pair_axes=None):
    return performance_figure(result, metric, full_screen, class_index, pair_axes)


def instance():
    this = Card(file=__file__, mutable=True, pipeline_compatible=True)
    this.long_name = 'Candidate ensemble'
    this.description = 'Combine weighted method winners and compare their resampling performance.'
    # These controls also drive evaluation while export is disabled. Acknowledge
    # their initial/browser-updated values before later user edits, otherwise
    # enabling export can wait forever for an obsolete default or bookmark.
    this.export_inputs('Apply', 'Models', 'Style')
    this.register_configuration_state_field('evaluation')
    this.exclude_configuration_input('View')

    this.front = lambda: ui.div(
        shinywidgets.output_widget(
            id='Chart', fill=True, guide=this, title='Ensemble resampling statistics',
            text='Lower triangle: residual correlations. Upper triangle: matched held-out residual pairs. '
                 'Bars show resampled performance. These post-selection results are not independent test performance.',
            position='left'),
        class_='ensemble-chart html-fill-container html-fill-item'
    )

    this.back = lambda: ui.TagList(
        ui.span('Evaluation results', class_='text-primary text-center d-block'),
        ui.output_ui(id='Table')
    )

    this.footer = lambda: ui.TagList(
        ui.output_ui(id='Busy'),
        ui.input_checkbox_group(
            id='Apply', label=None, choices=['Ensemble'], selected=[], inline=True,
            guide=this, title='Use ensemble', text='Export the ensemble definition and split-fitted evaluation models. '
                'Final fitting remains a downstream task.', position='top'
        ),
        # ui.output_text(id='Resources'),  #TODO: Needed or not
        ui.output_text(id='Status')
    )

    this.settings = lambda: ui.TagList(
        ui.input_checkbox_group(id='Models', label='Member models', choices={}),
        ui.input_select(id='Style', label='Ensemble style', choices=STYLES['regression'], selected='mean'),
        ui.input_select(id='Metric', label='Preferred metric', choices=METRICS['regression']),
        ui.input_select(id='ResidualView', label='Residual diagnostic', choices={'0': 'Regression residuals'})
    )

    def server(input, output, session):
        fit_service = session_fit_service(session)
        if fit_service.runtime is not None:
            fit_service.enable(this.namespace)
            session.on_destroy(lambda: fit_service.remove_consumer(this.namespace))
        busy = this.busy()
        chart_widget = WidgetSlot(session)
        chart_instance = reactive.Value(None)
        pair_axes = {}  # Stable numeric subplot identities across membership/ranking changes.
        residual_choices_identity = None
        restored_residual_view = this.restored_configuration_input('ResidualView', '0', warn_if_missing=False)
        cancellation = None
        requested = reactive.Value(None)
        completed = reactive.Value(None)
        evaluation_context = reactive.Value(None)
        latest_signature = reactive.Value(None)
        waiting = reactive.Value('')
        choices_identity = None
        restored_models = this.restored_configuration_input('Models', None, warn_if_missing=False)
        restored_style = this.restored_configuration_input('Style', None, warn_if_missing=False)
        restored_metric = this.restored_configuration_input('Metric', None, warn_if_missing=False)
        try:
            restored_display = restore_ensemble_display(this.consume_restored_configuration_field('evaluation', None))
        except ValueError as exc:
            this.log.warning(str(exc))
            restored_display = None
        saved_display = reactive.Value(restored_display)
        this.register_configuration_state_provider('evaluation', lambda: (
            portable_ensemble_display(saved_display()) if saved_display() is not None else None))

        @this.reactable(calc=True)
        @this.record_context
        def Incoming():
            try:
                source = this.input_data()
            except SilentException:
                # A temporarily unpublished upstream snapshot is not a new
                # selection: retain any evaluation already in progress.
                req(False)
            req(source is not None)
            return source

        @this.reactable(calc=True)
        def Available():
            try:
                return representatives(Incoming()), ''
            except ValueError as exc:
                return None, str(exc)

        @this.reactable()
        def Choices():
            nonlocal choices_identity
            available, _ = Available()
            req(available is not None)
            results, members = available
            task = results['state']['specification']['task']
            identity = (task, tuple(members))
            if identity == choices_identity:
                return
            with reactive.isolate():
                selected = input.Models() if choices_identity is not None else restored_models
                style = input.Style() if choices_identity is not None else restored_style
                metric = input.Metric() if choices_identity is not None else restored_metric
            # New upstream members are selected by default; preserve deliberate exclusions.
            if selected is None:
                selected = list(members)
            else:
                selected = [key for key in selected if key in members] + [
                    key for key in members if choices_identity is not None and key not in choices_identity[1]]
            ui.update_checkbox_group('Models', choices={key: value['label']
                for key, value in members.items()}, selected=selected)
            ui.update_select('Style', choices=STYLES[task], selected=style if style in STYLES[task] else next(iter(STYLES[task])))
            ui.update_select('Metric', choices=METRICS[task], selected=metric if metric in METRICS[task] else next(iter(METRICS[task])))
            choices_identity = identity

        @this.reactable(calc=True)
        @this.settle(seconds=0.5)
        def Options():
            selected, style = input.Models(), input.Style()
            req(selected is not None, style is not None)
            return tuple(selected), style

        @this.reactable(calc=True)
        @this.settle(seconds=0.5)
        def Enabled():
            return 'Ensemble' in (input.Apply() or [])

        @busy.track('Evaluating member models and ensemble…')
        @this.extended_task
        @this.record_context
        async def Evaluate(source, options, signature):
            nonlocal cancellation
            stop = threading.Event()
            cancellation = stop
            try:
                result = evaluate_ensemble(source, *options, cancelled=stop, fit_service=fit_service) if Module.IS_SHINYLIVE else await asyncio.to_thread(
                    evaluate_ensemble, source, *options, cancelled=stop, fit_service=fit_service)
                return signature, result, ''
            except asyncio.CancelledError:
                stop.set()
                raise
            except Exception as exc:  # Provider/worker failures belong in the card status.  # noqa: BLE001
                return signature, None, f'{type(exc).__name__}: {exc}'

        @this.reactable()
        def Start():
            available, error = Available()
            try:
                if available is None:
                    raise ValueError(error)
                source, options = Incoming(), Options()
                signature = evaluation_signature(source, *options)
                context = evaluation_context_signature(source)
            except ValueError as error:
                with reactive.isolate():
                    had_snapshot = requested() is not None
                if had_snapshot:
                    if cancellation is not None:
                        cancellation.set()
                    Evaluate.cancel()
                    requested.set(None)
                    completed.set(None)
                    if fit_service.runtime is not None:
                        fit_service.publish_snapshot(this.namespace, None)
                latest_signature.set(None)
                waiting.set(str(error))
                return
            waiting.set('')
            with reactive.isolate():
                if latest_signature() != signature:
                    latest_signature.set(signature)
                if evaluation_context() != context:
                    evaluation_context.set(context)
            finished = completed()  # Completion wakes scheduling of the latest snapshot.
            with reactive.isolate():
                saved = saved_display()
            if finished is None and saved is not None and saved.input_signature == signature:
                completed.set((signature, context, options, saved, ''))
                return
            if (finished is not None and finished[0] == signature
                    and isinstance(finished[3], EnsembleDisplay)
                    and finished[3].residuals is not None and not Enabled()):
                # Complete saved diagnostics restore without runtime fits. Older
                # score-only evaluations stay visible while their missing
                # diagnostics are acquired, even with ensemble export disabled.
                return
            with reactive.isolate():
                previous = requested()
                status = Evaluate.status()
            if previous is not None and previous[2] == signature and status != 'cancelled':
                return
            if previous is not None and previous[1] == options and previous[3] == context and status == 'running':
                # Winner changes are coalesced: let this snapshot finish first.
                return
            if finished is not None and (finished[1] != context or finished[2] != options):
                completed.set(None)
                if fit_service.runtime is not None:
                    fit_service.publish_snapshot(this.namespace, None)
            if cancellation is not None:
                cancellation.set()
            Evaluate.cancel()
            requested.set((source, options, signature, context))
            Evaluate.invoke(source, options, signature)

        @this.reactable()
        def Collect():
            signature, result, error = Evaluate.result()
            snapshot = requested()
            req(snapshot is not None)
            req(snapshot[2] == signature)
            completed.set((signature, snapshot[3], snapshot[1], result, error))
            if result is not None:
                saved_display.set(restore_ensemble_display(portable_ensemble_display(result)))
            if fit_service.runtime is not None:
                fit_service.publish_snapshot(this.namespace, result.cache_owner if result else None)

        @this.reactable(calc=True)
        def Results():
            if waiting():
                return None, waiting()
            finished = completed()
            req(finished is not None)
            req(finished[1] == evaluation_context(), finished[2] == Options())
            return finished[3], finished[4]

        @output
        @render_widget
        @this.record_context
        def Chart():
            widget = chart_widget.replace(go.FigureWidget(Card.empty_figure('Waiting for ensemble evaluation.')))
            widget._config = {'displayModeBar': False, 'displaylogo': False, 'responsive': True}
            chart_instance.set(widget)
            return widget

        @this.reactable()
        def ResidualChoices():
            nonlocal residual_choices_identity
            result, _ = Results()
            req(result is not None)
            classes = tuple(result.residuals.summary['classes']) if result.residuals is not None else ()
            if classes == residual_choices_identity:
                return
            choices = {str(i): f'Class {label} probability error' for i, label in enumerate(classes)} if classes else {'0': 'Regression residuals' if result.task == 'regression' else 'Probability residuals unavailable'}
            with reactive.isolate():
                value = input.ResidualView() if residual_choices_identity is not None else restored_residual_view
            ui.update_select('ResidualView', choices=choices, selected=value if value in choices else next(iter(choices)))
            residual_choices_identity = classes

        @this.reactable(calc=True)
        def ResidualIndex():
            value = str(input.ResidualView() or '0')
            return int(value) if value.isdecimal() else 0

        @this.reactable()
        @this.record_context
        async def UpdateChart():
            req(this.isFront())
            widget = chart_instance()
            req(widget is not None)
            full = bool(this.isFullScreen())
            try:
                result, error = Results()
                metric = input.Metric()
                figure = _figure(result, metric, full, ResidualIndex(), pair_axes) if result and metric in METRICS[result.task] else Card.empty_figure(error or 'Choose a metric.')
            except SilentException:
                # Keep the last rendered snapshot while a valid replacement is
                # evaluating. Replacing its multi-axis layout with a placeholder
                # mid-update leaves Plotly's subplot domains inconsistent.
                for trace in widget.data:
                    if trace.meta and trace.meta.get('role') == 'correlation':
                        trace.showscale = full
                return
            # Keep unreferenced numeric axes when the selected matrix shrinks.
            # Removing them before the widget's trace update makes Plotly try
            # to draw old pair traces against axes that no longer exist.
            next_layout = figure.layout.to_plotly_json()
            figure.update_layout({key: value for key, value in widget.layout.to_plotly_json().items()
                                  if key.startswith(('xaxis', 'yaxis')) and key not in next_layout})
            # A restored density becomes a scatter after runtime evaluation.
            # Plotly trace types cannot be changed through an in-place restyle.
            wanted_types = {trace.legendgroup or trace.name: trace.type for trace in figure.data}
            kept = tuple(trace for trace in widget.data
                         if trace.type == wanted_types.get(trace.legendgroup or trace.name, trace.type))
            if len(kept) != len(widget.data):
                widget.data = kept
            config = {'displayModeBar': full, 'displaylogo': False, 'responsive': True}
            if update_figure(widget, figure, config, preserve_traces=True):
                await session.send_custom_message('update_plotly_config', {'id': this.ns('Chart'), 'config': config})

        @output
        @render.ui
        def Table():
            req(Incoming())
            # Recreate on source changes: moving a card reconnects Shiny's
            # custom element, which can append a second internal grid.
            return ui.output_data_frame(id = "Table2")

        @output
        @render.data_frame
        def Table2():
            result, error = Results()
            metric = input.Metric()
            if result:
                req(metric in METRICS[result.task])
            table = result.metric_table(metric) if result else pd.DataFrame({'Note': [error]})
            return render.DataTable(table.round(3), width='100%', height=None)

        @output
        @render.ui
        def Busy():
            # This task owns a captured input snapshot and keeps running through
            # transient upstream/bookmark waits. Its spinner must do the same.
            return busy.ui(suppress_when_waiting=False)

        @output
        @render.text
        def Resources():
            if fit_service.runtime is None or Evaluate.status() != 'running':
                return ''
            reactive.invalidate_later(.5)
            resources = fit_service.runtime.snapshot()
            if resources['training_searches']:
                return 'Training has priority; ensemble evaluation uses spare capacity.'
            return ''

        @output
        @render.text
        def Status():
            result, error = Results()
            if error:
                return error
            origin = result.evaluation['origin']
            evidence, winners = Available()[0]
            winner_trial = evidence.get('review', {}).get('winner_trial')
            model_name = next((trial.get('method', trial['model_id'])
                for trial in evidence['state']['trials'] if trial['id'] == winner_trial), 'selected model')
            excluded = [c['label'] for c in Incoming().model_candidates if c['id'] not in winners]
            note = ('No eligible weighted winner: ' + ', '.join(excluded) + '. ') if excluded else ''
            finished = completed()
            if finished is not None and finished[0] != latest_signature():
                note += 'Showing an earlier evaluated snapshot; newer winners are being evaluated. '
            diagnostics = result.diagnostics if isinstance(result, EnsembleEvaluation) else {}
            if isinstance(result, EnsembleDisplay):
                note += 'Saved evaluation scores. '
            else:
                note += f"{diagnostics.get('reused_folds', 0)} fold fits reused; {diagnostics.get('new_fits', 0)} new. "
            if diagnostics.get('serialization_fallbacks'):
                note += 'Some providers required local fitting. '
            if fit_service.runtime is not None:
                resources = fit_service.runtime.snapshot()
                if resources['pinned_bytes'] > fit_service.runtime.cache_budget:
                    note += f"Current fitted models use {resources['pinned_bytes'] / 1024**3:.1f} GiB outside the evictable cache allowance. "
            return note + f'{len(result.members)} members · {result.split_count} {origin} splits · ' + (
                ('Ensemble model passed through'
                 if finished[0] == latest_signature() and this.export_readiness.phase() is None
                 else 'Preparing current ensemble for downstream use…')
                if Enabled() else f"Evaluation only; model '{model_name}' passed through")

        @this.reactable(calc=True)
        def Export():
            if not Enabled():
                source = Incoming()
                req(source is not None)
                return source
            result, _ = Results()
            snapshot = requested()
            req(snapshot is not None, isinstance(result, EnsembleEvaluation),
                result.input_signature == latest_signature(), snapshot[2] == result.input_signature)
            # Irrelevant training publications must not repeatedly invalidate an
            # already-ready ensemble export. This is the evaluated source;
            # Start invalidates readiness when its fitting identity changes.
            return snapshot[0].with_ensemble(result)

        @this.export_check
        def export_ready():
            this.require_settled(Enabled)
            if Enabled():
                this.require_settled(Options)
                snapshot = requested()
                req(snapshot is not None)
                _source, options, signature, _ = snapshot
                req(options == Options(), signature == latest_signature())
                # Collect owns completed snapshots independently of the task.
                # Comparing the task's whole proxy argument also traverses the
                # changing upstream search history; only fitting identity is a
                # prerequisite for exporting this evaluated ensemble.
                result, error = Results()
                if error:
                    raise ValueError(error)
                req(isinstance(result, EnsembleEvaluation), result.input_signature == signature)

        def close():
            if cancellation is not None:
                cancellation.set()
            Evaluate.cancel()
        session.on_ended(close)
        return Export

    this.server = server
    return this


if Module.running_directly(name=__name__):
    from ensemble_scenario import example_source
    this = instance()
    this._imports.set(example_source())
    this.run()
