"""Evaluate ensembles of weighted method winners without final fitting."""
import asyncio
import threading

import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from code_recording import recordable
from ensemble import evaluate_ensemble, evaluation_signature, representatives
from module import Module
from pipeline_search import METRICS
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget
from widget_lifecycle import WidgetSlot, update_figure

STYLES = {'regression': {'mean': 'Equal mean', 'median': 'Median'},
          'classification': {'soft': 'Soft voting (mean probabilities)', 'hard': 'Hard majority voting'}}


@recordable
def _figure(result, metric, full_screen=False):
    table = result.table(metric)
    figure = go.Figure(go.Bar(x=table.Score, y=table.Model, orientation='h',
        marker_color=['#e69f00' if key == 'ensemble' else '#337ab7' for key in table.Key],
        customdata=table[['SD', 'Splits']],
        hovertemplate='%{y}<br>Score %{x:.5g}<br>Fold SD %{customdata[0]:.5g}<br>%{customdata[1]} splits<extra></extra>'))
    figure.update_layout(xaxis_title=METRICS[result.task][metric], yaxis={'autorange': 'reversed'},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#bbd6f8', showlegend=False,
        margin={'l': 15, 'r': 15, 't': 10, 'b': 35}, font={'size': 13 if full_screen else 10})
    return figure


def instance():
    this = Card(file=__file__, mutable=True, pipeline_compatible=True)
    this.long_name = 'Candidate ensemble'
    this.description = 'Combine weighted method winners and compare their resampling performance.'
    this.export_inputs('Apply')
    
    this.front = lambda: ui.TagList(
        ui.span('Post-selection resampling performance', class_='text-primary text-center d-block'),
        shinywidgets.output_widget(
            id='Chart', fill=True, guide=this, title='Ensemble performance',
            text='Members and ensemble use identical splits. Orange marks the ensemble; bars run best to worst. '
                 'These post-selection results are not independent test performance.', position='left'
        )
    )
    
    this.back = lambda: ui.TagList(
        ui.span('Evaluation results', class_='text-primary text-center d-block'),
        ui.output_data_frame(id='Table')
    )
    
    this.footer = lambda: ui.TagList(
        ui.output_ui(id='Busy'), 
        ui.output_text(id='Status'),
        ui.input_checkbox_group(
            id='Apply', label=None, choices=['Ensemble'], selected=[], inline=True,
            guide=this, title='Use ensemble', text='Export the ensemble definition and split-fitted evaluation models. '
                'Final fitting remains a downstream task.', position='top'
        )
    )

    this.settings = lambda: ui.TagList(
        ui.input_checkbox_group(id='Models', label='Member models', choices={}),
        ui.input_select(id='Style', label='Ensemble style', choices=STYLES['regression'], selected='mean'),
        ui.input_select(id='Metric', label='Preferred metric', choices=METRICS['regression'])
    )

    def server(input, output, session):
        busy = this.busy()
        chart_widget = WidgetSlot(session)
        chart_instance = reactive.Value(None)
        cancellation = None
        requested = reactive.Value(None)
        waiting = reactive.Value('')
        choices_identity = None
        restored_models = this.restored_configuration_input('Models', None, warn_if_missing=False)
        restored_style = this.restored_configuration_input('Style', None, warn_if_missing=False)
        restored_metric = this.restored_configuration_input('Metric', None, warn_if_missing=False)
        this.export_inputs('Models', when=lambda: 'Ensemble' in (input.Apply() or []))
        this.export_inputs('Style', when=lambda: 'Ensemble' in (input.Apply() or []))

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
            identity = (task, tuple(members), tuple(value['trial'] for value in members.values()))
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
            ui.update_checkbox_group('Models', choices={key: f"{value['label']} · Trial {value['trial'] + 1}"
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
                result = evaluate_ensemble(source, *options, cancelled=stop) if Module.IS_SHINYLIVE else await asyncio.to_thread(
                    evaluate_ensemble, source, *options, cancelled=stop)
                return signature, result, ''
            except (ValueError, TypeError, InterruptedError) as exc:
                return signature, None, str(exc)

        @this.reactable()
        def Start():
            source, options = Incoming(), Options()
            try:
                signature = evaluation_signature(source, *options)
            except ValueError as error:
                if cancellation is not None:
                    cancellation.set()
                Evaluate.cancel()
                requested.set(None)
                waiting.set(str(error))
                return
            waiting.set('')
            with reactive.isolate():
                previous = requested()
            if previous is not None and signature is not None and previous[2] == signature:
                return
            if cancellation is not None:
                cancellation.set()
            Evaluate.cancel()
            requested.set((source, options, signature))
            Evaluate.invoke(source, options, signature)

        @this.reactable(calc=True)
        def Results():
            if waiting():
                return None, waiting()
            try:
                signature, result, error = Evaluate.result()
            except SilentOperationInProgressException:
                req(False)
            snapshot = requested()
            req(snapshot is not None, signature == snapshot[2], snapshot[1] == Options())
            return result, error

        @output
        @render_widget
        @this.record_context
        def Chart():
            widget = chart_widget.replace(go.FigureWidget(Card.empty_figure('Waiting for ensemble evaluation.')))
            widget._config = {'displayModeBar': False, 'displaylogo': False, 'responsive': True}
            chart_instance.set(widget)
            return widget

        @this.reactable()
        @this.record_context
        async def UpdateChart():
            widget = chart_instance()
            req(widget is not None)
            full = bool(this.isFullScreen())
            try:
                result, error = Results()
                metric = input.Metric()
                figure = _figure(result, metric, full) if result and metric in METRICS[result.task] else Card.empty_figure(error or 'Choose a metric.')
            except SilentException:
                figure = Card.empty_figure(Available()[1] or 'Waiting for ensemble evaluation.')
            config = {'displayModeBar': full, 'displaylogo': False, 'responsive': True}
            if update_figure(widget, figure, config):
                await session.send_custom_message('update_plotly_config', {'id': this.ns('Chart'), 'config': config})

        @output
        @render.data_frame
        def Table():
            result, error = Results()
            metric = input.Metric()
            if result:
                req(metric in METRICS[result.task])
            table = result.metric_table(metric) if result else pd.DataFrame({'Note': [error]})
            return render.DataTable(table, width='100%', height=None)

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render.text
        def Status():
            if Available()[1]:
                return Available()[1]
            result, error = Results()
            if error:
                return error
            origin = result.evaluation['origin']
            winners = Available()[0][1]
            excluded = [c['label'] for c in Incoming().model_candidates if c['id'] not in winners]
            note = ('No eligible weighted winner: ' + ', '.join(excluded) + '. ') if excluded else ''
            return note + f'{len(result.members)} members · {len(result.splits)} {origin} splits · ' + (
                'Ensemble enabled; final fitting is downstream.' if Enabled() else 'Evaluation only; incoming model passed through.')

        @this.reactable(calc=True)
        def Export():
            source = Incoming()
            if not Enabled():
                return source
            result, _ = Results()
            req(result is not None)
            return source.with_ensemble(result)

        @this.export_check
        def export_ready():
            this.require_settled(Enabled)
            if Enabled():
                this.require_settled(Options)
                snapshot = requested()
                req(snapshot is not None)
                source, options, signature = snapshot
                req(options == Options(), signature == evaluation_signature(Incoming(), *options))
                Evaluate.require_current(source, options, signature)

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
