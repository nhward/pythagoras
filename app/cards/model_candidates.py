"""Review predictive trade-offs among completed tuned models and choose one."""
from copy import deepcopy

import joblib
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from faicons import icon_svg as icon
from model_review import (
    axis_options,
    error_figure,
    parameter_differences,
    representatives,
    review_frame,
    tradeoff_figure,
    within_figure,
)
from pipeline_search import METRICS
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget
from widget_lifecycle import WidgetSlot, update_figure


def instance():
    this = Card(file=__file__, mutable=True)
    this.register_configuration_state_field('review')
    this.long_name = 'Model candidates'
    this.description = 'Compare tuned configurations and model methods using performance, error risk and observed prediction speed.'
    this.front = lambda: ui.TagList(
        ui.input_select('Method', label='Method', choices={}),
        shinywidgets.output_widget('ReviewPlot', fill=True, guide=this, title='Model trade-offs',
            text='Choose a view in settings. Equivalence is relative to each method’s best. '
                 'Trade-off diamonds identify the Pareto frontier on the displayed axes. '
                 'Use the flip-side table to choose representatives or accept a model. All comparisons are exploratory.'),
        ui.output_text('Status'),
    )
    this.back = lambda: ui.TagList(
        ui.output_text('Note'),
        ui.output_data_frame('Candidates'),
        ui.tags.details(ui.tags.summary('Selected configuration and parameter differences'),
                        ui.output_text('Details'),
                        ui.input_select('CompareWith', label='Compare selected configuration with', choices={}),
                        ui.output_data_frame('Differences'), id=this.ns('ComparisonDetails'),
                        class_='model-candidates-differences'),
        ui.download_button('ResultsCSV', label='Download', width='220px',
                           icon=icon('file-arrow-down', a11y='deco'),
                           class_='btn rounded-pill btn-sm d-block mx-auto btn-primary',
                           style='border: 0px; box-shadow: none;'),
    )
    this.footer = lambda: ui.TagList(
        ui.input_action_button('Representative', label='Use selected as representative', class_='btn rounded-pill btn-sm btn-primary', style='border: 0px; box-shadow: none;'),
        ui.input_action_button('Accept', label='Accept selected model', class_='btn rounded-pill btn-sm btn-primary', style='border: 0px; box-shadow: none;'),
        ui.input_action_button('Revert', label='Revert acceptance', class_='btn rounded-pill btn-sm btn-primary', style='border: 0px; box-shadow: none;'),
        ui.output_text('Message'),
    )

    def settings():
        return ui.TagList(
            ui.input_radio_buttons('View', label='View', choices={'within': 'Within method', 'tradeoff': 'Between methods', 'errors': 'Error exceedance'}, selected='within',
                guide=this, title='Review view', position='left', text='Inspect nearby configurations, compare one representative per method, or inspect held-out error tails.'),
            ui.input_numeric('EquivalenceTolerance', label='Practical-equivalence tolerance (metric units)', value=None, min=0,
                guide=this, title='Equivalence tolerance', position='left', text='Choose a positive application-based tolerance, ideally before reviewing results. Required to assess practical equivalence, not to display scores.'),
            ui.input_numeric('ComparisonLimit', label='Top trials per method', value=5, min=1, max=20, step=1,
                guide=this, title='Per-method shortlist', position='left', text='Each method retains its own highest-scoring trials. Inconclusive trials remain visible; they are not declared equivalent.'),
            ui.input_numeric('DecisionProbability', label='Comparison decision probability', value=.95, min=.501, max=.999, step=.01,
                guide=this, title='Decision probability', position='left', text='Posterior probability required to label practical equivalence or meaningful inferiority. Comparisons require upstream repeated shuffled ordinary or stratified CV.'),
            ui.input_select('XAxis', label='Between-method x-axis', choices={'Score': 'Score'}, selected='Score',
                guide=this, title='Trade-off x-axis', position='left', text='Choose a numerical criterion. Score follows the tuning metric’s direction.'),
            ui.input_select('YAxis', label='Between-method y-axis', choices={'P95 absolute error': 'P95 absolute error'},
                guide=this, title='Trade-off y-axis', position='left', text='Regression tail risk is the mean of each repeat’s 95th absolute-error percentile. Classification uses error rate or log loss. Observed rows/s is warm validation-batch throughput, not a production benchmark.'),
        )
    this.settings = settings
    this.exclude_configuration_input('CompareWith')

    def server(input, output, session):
        review_widget = WidgetSlot(session)
        plot_instance = reactive.Value(None)
        saved = this.restored_configuration_field('review', {})
        saved = saved if isinstance(saved, dict) else {}
        choices = reactive.Value(saved.get('representatives') if isinstance(saved.get('representatives'), dict) else {})
        fingerprint = reactive.Value(saved.get('fingerprint'))
        accepted = reactive.Value(saved.get('accepted') if type(saved.get('accepted')) is int and saved['accepted'] > 0 else None)
        message = reactive.Value('')
        selected_trial = reactive.Value(None)
        results_state = reactive.Value(None)
        results_source = reactive.Value(None)
        last_source = None
        last_revision = None
        configured_choices = {}
        restoring_selection = False
        this.register_configuration_state_provider('review', lambda: {
            'representatives': deepcopy(choices()), 'fingerprint': fingerprint(), 'accepted': accepted()})

        @this.reactable(calc=True)
        def Incoming():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(source is not None)
            return source

        @this.reactable()
        def PublishResults():
            nonlocal last_source, last_revision
            try:
                source = Incoming()
            except SilentException:
                # Keep the diagnostic view during transient export-readiness
                # changes. Exported() still requires the live upstream source.
                return
            if source is last_source:
                return
            results = source.tuning_results
            revision = joblib.hash(results)
            last_source = source
            if revision != last_revision:
                last_revision = revision
                results_state.set(results)
            results_source.set(source)

        @reactive.calc
        def Results():
            return results_state()

        @reactive.effect
        def Configure():
            results = Results()
            if results is None:
                return
            state = results['state']
            identity = state['specification']['fingerprint']
            with reactive.isolate():
                if fingerprint() != identity:
                    choices.set({}); accepted.set(None); selected_trial.set(None); fingerprint.set(identity)
                method = input.Method() or this.restored_configuration_input('Method', '')
                xaxis, yaxis = input.XAxis(), input.YAxis()
            methods = {t['model_id']: t.get('method', t['model_id']) for t in state['trials']}
            if methods != configured_choices.get('Method') or method not in methods:
                ui.update_select('Method', choices=methods, selected=method if method in methods else next(iter(methods), ''))
                configured_choices['Method'] = methods
            axes = axis_options(state['specification']['task'])
            for name, value, default in (('XAxis', xaxis, axes[0]), ('YAxis', yaxis, axes[1])):
                if axes != configured_choices.get(name) or value not in axes:
                    ui.update_select(name, choices=axes, selected=value if value in axes else default)
                    configured_choices[name] = axes

        @reactive.calc
        def Review():
            limit = input.ComparisonLimit()
            return review_frame(Results(), input.EquivalenceTolerance(),
                                limit=int(limit) if limit is not None and float(limit).is_integer() else 0,
                                probability=.95 if input.DecisionProbability() is None else input.DecisionProbability())

        @reactive.calc
        def Representatives():
            return representatives(Review()[0], choices())

        @output(suspend_when_hidden=False)
        @render.data_frame
        def Candidates():
            nonlocal restoring_selection
            with reactive.isolate():
                restoring_selection = selected_trial() is not None
            frame = Review()[0].copy()
            if not frame.empty:
                frame.insert(2, 'Representative', frame.Trial.isin(Representatives().Trial))
            styles = [] if frame.empty else [{
                'rows': [i for i, assessment in enumerate(frame.Assessment) if assessment == 'Best score (reference)'],
                'class': 'model-candidates-reference',
            }]
            return render.DataGrid(frame.round(3), selection_mode='row', height='350px', filters=False,
                                   styles=styles)

        @reactive.effect
        def RememberSelection():
            nonlocal restoring_selection
            try:
                selected = Candidates.data_view(selected=True)
            except (SilentOperationInProgressException, IndexError):
                return
            if not selected.empty:
                restoring_selection = False
                selected_trial.set(int(selected.iloc[0]['Trial']))
            elif not restoring_selection:
                selected_trial.set(None)

        @reactive.calc
        def Selected():
            return selected_trial()

        @reactive.effect
        async def OpenSelectedDetails():
            trial, frame = Selected(), Review()[0]
            await session.send_custom_message('set_details_open', {
                'id': this.ns('ComparisonDetails'),
                'open': trial is not None and not frame.empty and trial in frame.Trial.values,
            })

        @reactive.effect
        def ComparisonChoices():
            frame, selected = Review()[0], Selected()
            if frame.empty or selected is None or selected not in frame.Trial.values:
                return
            key = frame.loc[frame.Trial == selected].iloc[0].Model
            group = frame.loc[frame.Model == key]
            options = {str(int(row.Trial)): f'Trial {row.Trial} · Score {row.Score:.6g}' for _, row in group.iterrows()}
            with reactive.isolate(): previous = input.CompareWith()
            default = next((str(int(t)) for t in group.Trial if t != selected), str(selected))
            ui.update_select('CompareWith', choices=options, selected=previous if previous in options and previous != str(selected) else default)

        @reactive.effect
        async def HighlightSelection():
            nonlocal restoring_selection
            trial = selected_trial()
            if trial is None:
                return
            frame = Review()[0]
            indices = [i for i, value in enumerate(frame.Trial) if value == trial] if not frame.empty else []
            if not indices:
                restoring_selection = False
                selected_trial.set(None)
                return
            try:
                table = Candidates.data()
            except (SilentOperationInProgressException, RuntimeError, IndexError):
                return
            restoring_selection = True

            async def restore():
                nonlocal restoring_selection
                with reactive.isolate():
                    if selected_trial() != trial or Candidates.data() is not table:
                        return
                # Send after the new grid, so its reset cannot erase selection.
                await Candidates.update_cell_selection({'type': 'row', 'rows': indices})
                restoring_selection = False

            session.on_flushed(restore, once=True)

        @output
        @render_widget
        def ReviewPlot():
            widget = review_widget.replace(go.FigureWidget())
            widget._config = {'displayModeBar': False, 'displaylogo': False, 'responsive': True}
            plot_instance.set(widget)
            return widget

        @reactive.effect
        async def UpdateReviewPlot():
            widget = plot_instance()
            req(widget is not None)
            results, frame = Results(), Review()[0]
            if results is None or frame.empty:
                figure = Card.empty_figure(Review()[1])
            elif input.View() == 'tradeoff':
                axes = axis_options(results['state']['specification']['task'])
                x = input.XAxis() if input.XAxis() in axes else axes[0]
                y = input.YAxis() if input.YAxis() in axes else axes[1]
                figure = tradeoff_figure(Representatives(), (x, y), results['state']['specification']['metric'])
            elif input.View() == 'errors':
                figure = error_figure(results, Representatives())
                if not figure.data:
                    figure = Card.empty_figure('Prediction evidence is unavailable; rerun tuning to collect it.')
            else:
                method = input.Method() or frame.iloc[0].Model
                figure = within_figure(frame, method, input.EquivalenceTolerance())
            figure.update_layout(modebar={'orientation': 'v'}, showlegend=bool(this.isFullScreen()) and input.View() in {'tradeoff', 'errors'})
            config = {'displayModeBar': bool(this.isFullScreen()), 'displaylogo': False, 'responsive': True}
            if update_figure(widget, figure, config):
                await session.send_custom_message('update_plotly_config', {'id': this.ns('ReviewPlot'), 'config': config})
            def SelectPoint(trace, points, selector):
                if points.point_inds and trace.meta and 'trials' in trace.meta:
                    selected_trial.set(int(trace.meta['trials'][points.point_inds[0]]))
            for trace in widget.data:
                if trace.meta:
                    trace.on_click(SelectPoint)

        @output
        @render.text
        def Note():
            return Review()[1]

        @output
        @render.text
        def Status():
            frame = Review()[0]
            if frame.empty:
                return Review()[1]
            spec = Results()['state']['specification']
            if Results().get('read_only'):
                return f'{len(frame)} trials · {frame.Model.nunique()} methods · Historical review (read-only)'
            return f"{METRICS[spec['task']][spec['metric']]} · " + f'{len(frame)} trials · {frame.Model.nunique()} methods · ' + ('Model accepted' if accepted() is not None else 'No model accepted')

        @output
        @render.text
        def Details():
            frame = Review()[0]
            trial = Selected()
            if trial is None or frame.empty or trial not in frame.Trial.values:
                return 'Select a trial to compare parameters.'
            row = frame.loc[frame.Trial == trial].iloc[0]
            return f'Trial {trial} · {row.Method}'

        @output
        @render.data_frame
        def Differences():
            frame, results, trial = Review()[0], Results(), Selected()
            if results is None or frame.empty or trial is None or trial not in frame.Trial.values:
                return render.DataGrid(pd.DataFrame(), height='180px')
            key = frame.loc[frame.Trial == trial].iloc[0].Model
            group = frame.loc[frame.Model == key]
            comparison = input.CompareWith()
            reference = int(comparison) if comparison and comparison.isdigit() and int(comparison) in group.Trial.values else int(group.iloc[0].Trial)
            if trial == reference:
                return render.DataGrid(pd.DataFrame({'Note': ['The two selected configurations are identical.']}), height='180px')
            try:
                differences = parameter_differences(results, trial, reference)
            except (ValueError, KeyError):
                differences = pd.DataFrame({'Note': ['Historical pipeline unavailable in this runtime.']})
            return render.DataGrid(differences, width='100%', height='180px')

        @this.reactable(triggers=[input.Representative])
        def UseRepresentative():
            trial, frame = Selected(), Review()[0]
            if trial is None or trial not in frame.Trial.values:
                message.set('Select a candidate row first.'); return
            row = frame.loc[frame.Trial == trial].iloc[0]
            choices.set({**choices(), row.Model: trial})
            ui.update_select('Method', selected=row.Model)
            message.set(f'Trial {trial} is the representative for {row.Method}.')

        @this.reactable(triggers=[input.Accept])
        def Accept():
            trial = Selected()
            if trial is None:
                message.set('Select a completed candidate row first.'); return
            try:
                source = Incoming()
                req(source is results_source())
                source.with_selected_model(trial - 1)
                accepted.set(trial)
                message.set(f'Trial {trial} accepted.')
            except ValueError as exc:
                message.set(str(exc))

        @this.reactable(triggers=[input.Revert])
        def Revert():
            accepted.set(None)
            message.set('Acceptance reverted.')

        @output
        @render.text
        def Message():
            return message()

        @output
        @render.download_button(filename='model-candidates.csv')
        def ResultsCSV():
            yield Review()[0].to_csv(index=False)

        @this.reactable(calc=True)
        def Exported():
            source = Incoming()
            # A diagnostic snapshot may remain visible while upstream is busy,
            # but it must never be attached to a newly arrived source.
            req(source is results_source())
            results = Results()
            if results is not None:
                frame = Review()[0]
                reps = Representatives()
                review = {'version': 1, 'shortlisted_trials': [int(t) - 1 for t in frame.Trial] if not frame.empty else [],
                          'representatives': {row.Model: int(row.Trial) - 1 for _, row in reps.iterrows()},
                          'tolerance': input.EquivalenceTolerance(), 'decision_probability': input.DecisionProbability(),
                          'accepted_trial': accepted() - 1 if accepted() is not None else None}
                source = source.with_tuning_results({**results, 'review': review})
            if accepted() is not None and results is not None and fingerprint() == results['state']['specification']['fingerprint']:
                try:
                    return source.with_selected_model(accepted() - 1)
                except ValueError:
                    pass
            return source.with_inactive_step(stage='Learning', card='model_candidates', operation='Accept reviewed model')

        return Exported

    this.server = server
    return this


if __name__ == '__main__':
    import numpy as np
    from evaluation_plan import build_plan
    from model_commit import prepare_predictors
    from model_review import completed_evidence
    from pipeline_search import fresh, prepare, run
    from proxy_data import proxy_data
    from roles import RoleMap
    from sklearn.base import clone
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    from tuning import compile_spec, discover_source

    x = np.arange(60, dtype=float)
    source = proxy_data(pd.DataFrame({'x': x, 'y': 2 * x + 8 * np.sin(x / 3)}),
                        _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
    source = source.with_evaluation_plan(build_plan(
        source, strategy='kfold', objective='independent', folds=3, repeats=2), card='obs_resample')
    ridge = Pipeline([
        ('predictors', FunctionTransformer(prepare_predictors,
            kw_args={'columns': ['x'], 'preparation': 'numeric'})),
        ('scale', StandardScaler()),
        ('estimator', Ridge()),
    ])
    source = source.with_model_candidates([
        {'id': 'ridge', 'label': 'Ridge Regression', 'estimator': ridge},
        {'id': 'tree', 'label': 'Decision Tree Regression',
         'estimator': clone(ridge).set_params(estimator=DecisionTreeRegressor(random_state=42))},
    ])
    rows = discover_source(source)
    drafts = {}
    for row in rows:
        if row.get('candidate_id') == 'ridge' and row['path'].endswith('__alpha'):
            values = [.1, 1., 10.]
        elif row.get('candidate_id') == 'tree' and row['path'].endswith('__max_depth'):
            values = [2, 3, 5]
        else:
            continue
        drafts[row['path']] = dict(row['definition'], values=values, owner=row['owner'], active=True)
    source = source.with_tuning_spec(compile_spec(rows, drafts))
    problem = prepare(source, method='grid')
    state = run(problem, fresh(problem), trials=0, publish=lambda state: None, pause=lambda: False)
    source = source.with_tuning_results(completed_evidence(problem, state))
    this = instance()
    this.restore_configuration_state({'inputs': {'EquivalenceTolerance': 1., 'ComparisonLimit': 5}})
    this._imports.set(source)
    this.run()
