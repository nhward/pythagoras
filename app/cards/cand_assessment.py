"""Assess the current reviewed model using held-out resampling predictions."""
import asyncio
import threading

import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from candidate_assessment import (
    assess_candidate,
    assessment_figure,
    assessment_signature,
    assessment_table,
    group_masks,
    grouping_columns,
    selection,
)
from card import Card
from fit_service import session_fit_service
from module import Module
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget
from widget_lifecycle import WidgetSlot, update_figure


def instance():
    this = Card(file=__file__, mutable=False, pipeline_compatible=True)
    this.long_name = 'Candidate assessment'
    this.description = 'Assess the selected candidate or committed ensemble using held-out resampling predictions.'
    
    this.front = lambda: ui.TagList(
        ui.span(ui.output_text('Title', inline=True), class_='text-primary text-center d-block'),
        shinywidgets.output_widget(
            id='Chart', fill=True, guide=this, title='Prediction performance',
            text='Regression compares actual and predicted values with a 1:1 line and central 95% signed-error limits. '
                 'Classification shows ROC using probabilities, decision scores or hard-vote shares; labels alone use a confusion matrix. '
                 'These post-selection results are not independent test performance.', position='left'
        )
    )

    this.back = lambda: ui.TagList(
        ui.span('Overall and subgroup performance', class_='text-primary text-center d-block'),
        ui.output_ui(
            id='Table', guide=this, title='Performance statistics',
            text='Errors are predicted minus actual. Error variance describes their spread, not a model bias–variance decomposition. '
                 'Observations count unique rows; predictions include repeated validation appearances. AUC is averaged over usable splits.', position='left'
        )
    )
    
    this.footer = lambda: ui.TagList(
        ui.output_ui(id='Busy'), 
        ui.output_text(id='Status')
    )
    
    this.settings = lambda: ui.TagList(
        ui.input_select(
            id='Color', label='Color by', choices={'': 'None'}, guide=this,
            text='Choose a Stratifier or Sensitive variable. Numeric variables with more than five values use five quantile bins; '
                 'categorical variables retain the 12 most frequent levels plus Other. Missing values form a separate group.', position='left'
        ),
        ui.input_select(
            id='Symbol', label='Symbol by', choices={'': 'None'}, guide=this,
            text='Choose a Stratifier or Sensitive variable for point symbols or ROC subgroup markers. '
                 'The table shows intersections when two different variables are selected.', position='left'
        ),
        ui.input_select(
            id='ChartType', label='Classification chart', choices={'auto': 'ROC when available', 'confusion': 'Confusion matrix'}, guide=this,
            text='Confusion matrices show percentages within each actual class and counts in hover. This setting does not affect regression.', position='left'
        ),
        ui.input_select(
            id='Class', label='Positive / one-versus-rest class', choices={}, guide=this,
            text='Select the positive class for binary ROC or the class compared with all others for multiclass ROC. '
                 'Probability-error statistics use this class. Vote share is a ranking score, not a probability.', position='left'
        ),
        ui.input_select(
            id='Subgroup', label='Confusion-matrix group', choices={'Overall': 'Overall'}, guide=this,
            text='Choose the overall population or a group defined by Colour by and Symbol by. All groups remain in the table.', position='left'
        )
    )

    def server(input, output, session):
        service = session_fit_service(session)
        if service.runtime is not None:
            service.enable(this.namespace, assessment=True)
            session.on_destroy(lambda: service.remove_consumer(this.namespace) if not service._closed else None)
        busy = this.busy()
        slot = WidgetSlot(session)
        widget_value = reactive.Value(None)
        requested = reactive.Value(None)
        finished = reactive.Value(None)
        waiting = reactive.Value('Waiting for an upstream selected model.')
        cancellation = None
        group_identity = class_identity = subgroup_identity = None
        restored = {name: this.restored_configuration_input(name, '', warn_if_missing=False)
                    for name in ('Color', 'Symbol', 'Class', 'Subgroup')}

        @this.reactable(calc=True)
        def Incoming():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(source is not None)
            return source

        @output
        @render.text
        def Title():
            try:
                ensemble, _, trial = selection(Incoming())
            except ValueError:
                return 'Waiting for an upstream selected model'
            name = 'ensemble' if ensemble is not None else trial.get('method', trial['model_id'])
            return f'Assessment of {name} model'

        @busy.track('Collecting resampled validation predictions…')
        @this.extended_task
        @this.record_context
        async def Calculate(source, signature, stop):
            try:
                result = assess_candidate(source, fit_service=service, cancelled=stop) if Module.IS_SHINYLIVE else await asyncio.to_thread(
                    assess_candidate, source, fit_service=service, cancelled=stop)
                return signature, result, ''
            except asyncio.CancelledError:
                stop.set()
                raise
            except Exception as exc:  # Provider failures must remain visible.  # noqa: BLE001
                return signature, None, f'{type(exc).__name__}: {exc}'

        @this.reactable()
        def Start():
            nonlocal cancellation
            source = Incoming()
            try:
                signature = assessment_signature(source)
            except ValueError as exc:
                if cancellation is not None:
                    cancellation.set()
                Calculate.cancel()
                requested.set(None)
                finished.set(None)
                waiting.set(str(exc))
                return
            waiting.set('')
            with reactive.isolate():
                if requested() == signature and Calculate.status() != 'cancelled':
                    return
            if cancellation is not None:
                cancellation.set()
            Calculate.cancel()
            cancellation = threading.Event()
            requested.set(signature)
            finished.set(None)
            Calculate.invoke(source, signature, cancellation)

        @this.reactable()
        def Collect():
            signature, result, error = Calculate.result()
            req(signature == requested())
            finished.set((signature, result, error))

        @this.reactable(calc=True)
        def Results():
            value = finished()
            req(value is not None)
            req(value[0] == requested(), not waiting())
            req(value[0] == assessment_signature(Incoming()))
            return value[1], value[2]

        @this.reactable()
        def GroupChoices():
            nonlocal group_identity
            columns = tuple(grouping_columns(Incoming()))
            if columns == group_identity:
                return
            with reactive.isolate():
                for name in ('Color', 'Symbol'):
                    value = getattr(input, name)() if group_identity is not None else restored[name]
                    ui.update_select(name, choices={'': 'None', **{c: c for c in columns}}, selected=value if value in columns else '')
            group_identity = columns

        @this.reactable()
        def ClassChoices():
            nonlocal class_identity
            result, _ = Results()
            req(result is not None)
            labels = tuple(str(c) for c in result.classes)
            if labels == class_identity:
                return
            choices = {str(i): label for i, label in enumerate(labels)}
            with reactive.isolate():
                value = input.Class() if class_identity is not None else restored['Class']
            ui.update_select('Class', choices=choices, selected=value if value in choices else str(max(0, len(labels) - 1)))
            class_identity = labels

        @this.reactable(calc=True)
        def Display():
            result, _ = Results()
            req(result is not None)
            color = input.Color() or ''
            symbol = input.Symbol() or ''
            color = color if color in result.groups else ''
            symbol = symbol if symbol in result.groups else ''
            value = input.Class() or '0'
            index = int(value) if value.isdigit() and int(value) < len(result.classes) else max(0, len(result.classes) - 1)
            return color, symbol, index

        @this.reactable()
        def SubgroupChoices():
            nonlocal subgroup_identity
            result, _ = Results()
            req(result is not None)
            color, symbol, _ = Display()
            groups = tuple(label for label, _ in group_masks(result, color, symbol))
            if groups == subgroup_identity:
                return
            with reactive.isolate():
                value = input.Subgroup() if subgroup_identity is not None else restored['Subgroup']
            ui.update_select('Subgroup', choices=list(groups), selected=value if value in groups else 'Overall')
            subgroup_identity = groups

        @output
        @render_widget
        def Chart():
            widget = slot.replace(go.FigureWidget(Card.empty_figure('Waiting for prediction assessment.')))
            widget._config = {'displayModeBar': False, 'displaylogo': False, 'responsive': True}
            widget_value.set(widget)
            return widget

        @this.reactable()
        @this.record_context
        async def UpdateChart():
            widget = widget_value()
            req(widget is not None)
            full = bool(this.isFullScreen())
            try:
                result, error = Results()
                figure = assessment_figure(result, *Display(), chart=input.ChartType() or 'auto',
                    subgroup=input.Subgroup() or 'Overall', full_screen=full) if result else Card.empty_figure(error)
            except SilentException:
                figure = Card.empty_figure(waiting() or 'Collecting resampled validation predictions…')
            config = {'displayModeBar': full, 'displaylogo': False, 'responsive': True}
            if update_figure(widget, figure, config):
                await session.send_custom_message('update_plotly_config', {'id': this.ns('Chart'), 'config': config})

        @output
        @render.ui
        @this.record_context
        def Table():
            req(Incoming())
            # Recreate on source changes: moving a card reconnects Shiny's
            # custom element, which can append a second internal grid.
            return ui.output_data_frame(id = "Table2")

        @output
        @render.data_frame
        @this.record_context
        def Table2():
            result, error = Results()
            table = assessment_table(result, *Display()) if result else pd.DataFrame({'Note': [error]})
            return render.DataTable(table.round(5), width='100%', height=None)

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render.text
        def Status():
            if waiting():
                return waiting()
            try:
                result, error = Results()
            except SilentException:
                return 'Collecting resampled validation predictions…'
            if error:
                return error
            text = (f'{result.label} · {result.events.row.nunique()} observations · {len(result.events)} validation predictions · '
                    f'{result.splits} splits · {result.reused_fits} reused fits · {result.new_fits} new fits. '
                    'Post-selection resampling; not independent test performance.')
            if result.task == 'classification' and result.score_kind:
                text += f' ROC score: {result.score_kind.lower()}.'
                text += f' Selected class: {result.classes[Display()[2]]} versus rest.'
            if result.score_kind == 'Vote share':
                text += ' Vote share is not a calibrated probability.'
            if (input.Color() or input.Symbol()):
                text += ' Grouping uses up to five numeric quantile bins or 12 categorical levels plus Other; missing values are separate.'
            return text + (' ' + ' '.join(result.notes) if result.notes else '')

        def stop():
            if cancellation is not None:
                cancellation.set()
            Calculate.cancel()
        session.on_ended(stop)
        return Incoming

    this.server = server
    return this
