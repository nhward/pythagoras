"""Choose models using weighted score, speed and task-specific error evidence."""
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from candidate_selection import (
    candidate_frame,
    decisions,
    pareto_figure,
    transparent_candidates,
    weights_from_handles,
)
from card import Card
from estimator_catalogue import load_catalogue
from faicons import icon_svg as icon
from fit_service import session_fit_service
from model_review import parameter_differences
from module import Module
from proxy_data import proxy_data
from roles import RoleMap
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget
from widget_lifecycle import WidgetSlot, update_figure

#TODO: Review flip-side
#TODO: Reviw quarto

def instance():
    this = Card(file=__file__, mutable=True)
    this.register_configuration_state_field('selection')
    this.long_name = 'Candidate models'
    this.description = 'Choose weightings that yield the desired optimal model(s).'

    this.front = lambda: ui.TagList(
        ui.output_ui(
            id='Panels', class_='cand-models-panels'
        ),
        shinywidgets.output_widget(
            id='ReviewPlot', fill=True, guide=this, title='Model trade-offs',
            text='Rotate to inspect score, prediction throughput and P95 absolute error (regression) or error rate (classification). Bright points are nondominated; '
                 'the diamond is the weighted winner. Tabs change the view, not the exported Overall winner.'
        ),
        ui.div(
            ui.output_text(id='Status'), 
            class_='cand-models-status'
        ),
    )
    
    this.back = lambda: ui.TagList(
        ui.output_text(id='Note'),
        ui.output_data_frame(id='Candidates'),
        ui.tags.details(
            ui.tags.summary('Selected configuration and parameter differences'),
            ui.output_text(id='Details'),
            ui.input_select(id='CompareWith', label='Compare selected configuration with', choices={}),
            ui.output_data_frame(
                id='Differences'), id=this.ns('ComparisonDetails'),
                class_='model-candidates-differences'
            ),
        ui.download_button(
            id='ResultsCSV', label='Download', width='220px',
            icon=icon('file-arrow-down', a11y='deco'),
            class_='btn rounded-pill btn-sm d-block mx-auto btn-primary',
            style='border: 0px; box-shadow: none;'
        ),
    )
    
    this.footer = lambda: ui.TagList(
        ui.div(
            ui.input_slider(
                id='Weights', label=None, min=0, max=100,
                value=(100, 100), step=1, width='100%', drag_range=False, post="%",
                guide=this, title='Selection preferences', position='top',
                text='The three slider segments allocate Score, Throughput and prediction error weights (P95 absolute error for regression; error rate for classification). '
                     'Both handles at the right give Score 100%. Selection changes on release.'
            ),
            class_='cand-models-weights', style='width: calc(50% + 3rem);'
        ),
        ui.output_text('WeightSummary')
    )


    def settings():
        return ui.TagList(
            ui.input_checkbox(
                id='UsuallyTransparentOnly', label='Require Usually transparent methods', value=False,
                guide=this, title='Intrinsic transparency', position='left',
                text='Restrict selection to methods rated Usually transparent in the catalogue. '
                     'Conditional, usually opaque, unreviewed and unavailable classifications are excluded. '
                     'This concerns intrinsic method transparency, not SHAP or a guarantee that every fitted pipeline is understandable. '
                     'The table, chart and weighted winners use this constraint; training evidence is retained.'
            ),
            ui.input_numeric(
                id='ErrorFloor', label='Minimum error for weighted products', value=1e-12, min=1e-15,
                guide=this, title='Minimum error', position='left',
                text='Positive floor for RMSE, MAE and P95 in target units; dimensionless for 1 − R² and classification losses. '
                     'Values below this floor are treated equally. Changing units may require changing this floor.'
            ),
            ui.input_numeric(
                id='EquivalenceTolerance', label='Practical-equivalence tolerance (metric units)', value=None, min=0,
                guide=this, title='Practical equivalence', position='left',
                text='Optional application-based tolerance for comparisons against each method’s best score. This does not restrict weighted selection.'
            ),
            ui.input_numeric(
                id='DecisionProbability', label='Comparison decision probability', value=.95, min=.501, max=.999, step=.01,
                guide=this, title='Comparison probability', position='left',
                text='Posterior threshold for the comparison table. Requires suitable repeated shuffled CV.'
            ),
        )
    this.settings = settings
    this.exclude_configuration_input('CompareWith')
    this.exclude_configuration_input('MethodPanel')
    this.exclude_configuration_input('WeightCommit')
    this.exclude_configuration_input('EligibleMethods')  # Retired bookmark filter.

    def server(input, output, session):
        catalogue = load_catalogue()
        fit_service = session_fit_service(session)
        if fit_service.runtime is not None:
            fit_service.runtime.register_review('review:' + this.namespace)
            session.on_destroy(lambda: fit_service.runtime.remove_review('review:' + this.namespace))
        review_widget = WidgetSlot(session)
        plot_instance = reactive.Value(None)
        selected_trial = reactive.Value(None)
        results_state = reactive.Value(None)
        results_source = reactive.Value(None)
        winners = reactive.Value({'overall': None, 'methods': {}})
        restoring_selection = False
        camera = None
        last_source = None
        last_revision = None
        method_choices = reactive.Value({})
        initial_weights = this.restored_configuration_input('Weights', (100, 100), warn_if_missing=False)

        async def bind_weights():
            await session.send_custom_message('cand_models_weights', {
                'id': this.ns('Weights'), 'commit': this.ns('WeightCommit'), 'summary': this.ns('WeightSummary')})
        session.on_flushed(bind_weights, once=True)
        previous_fingerprint = None
        this.register_configuration_state_provider('selection', lambda: {
            'version': 1, 'overall': winners()['overall'], 'methods': winners()['methods']})

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

        @reactive.calc
        def AllCandidates():
            return candidate_frame(Results(), input.EquivalenceTolerance(), input.DecisionProbability())

        @reactive.effect
        def Configure():
            nonlocal previous_fingerprint
            results = Results()
            if results is None:
                return
            identity = results['state']['specification']['fingerprint']
            if identity != previous_fingerprint:
                selected_trial.set(None)
                previous_fingerprint = identity
            frame = EligibleFrame()
            methods = {} if frame.empty else dict(zip(frame.Model, frame.Method))
            with reactive.isolate(): previous = method_choices()
            if methods != previous:
                method_choices.set(methods)

        @reactive.calc
        def EligibleFrame():
            frame = AllCandidates()[0]
            if input.UsuallyTransparentOnly() and not frame.empty:
                frame = transparent_candidates(frame, Results(), catalogue)
            return frame.reset_index(drop=True)

        @reactive.calc
        def Weighting():
            try:
                try:
                    handles = input.WeightCommit()
                except SilentException:
                    handles = initial_weights
                return weights_from_handles(handles), float(input.ErrorFloor()), ''
            except (ValueError, TypeError) as exc:
                return None, None, str(exc)

        @reactive.calc
        def Decision():
            frame = EligibleFrame()
            weights, floor, error = Weighting()
            if error or frame.empty:
                return {'overall': None, 'methods': {}}, error
            try:
                metric = Results()['state']['specification']['metric']
                return decisions(frame, metric, weights, floor), ''
            except ValueError as exc:
                return {'overall': None, 'methods': {}}, str(exc)

        @reactive.effect
        def PublishDecision():
            decision = Decision()[0]
            with reactive.isolate(): previous = winners()
            if decision != previous:
                winners.set(decision)
            results = Results()
            if fit_service.runtime is not None and results is not None and not results.get('read_only'):
                trials = results['state']['trials']
                selected = {value - 1 for value in decision['methods'].values() if value is not None}
                fit_service.runtime.pin_review('review:' + this.namespace,
                    results['state']['specification']['fingerprint'],
                    [joblib.hash(t['params']) for t in trials if t['id'] in selected],
                    [joblib.hash(t['params']) for t in trials])

        @output
        @render.ui
        def Panels():
            # Only rebuild navigation when method identities change. The chart
            # lives outside these panels and keeps one widget and camera.
            methods = method_choices()
            with reactive.isolate():
                try:
                    current = input.MethodPanel()
                except SilentException:
                    current = None
            panels = [ui.nav_panel('Overall', value='overall')]
            panels.extend(ui.nav_panel(label, value='method:' + str(key)) for key, label in methods.items())
            valid = {'overall', *('method:' + str(key) for key in methods)}
            return ui.navset_bar(
                *panels, 
                id='MethodPanel', 
                selected=current if current in valid else 'overall',
                title=None, 
                padding=0, 
                fillable=False,
                navbar_options=ui.navbar_options(
                    collapsible=True, 
                    underline=True, 
                    bg='lightgrey',
                    class_=(
                        "nominal-navbar force-collapse"
                        if not this.isFullScreen() and len(panels) > this.config.get("settings", {}).get("max_collapse", 6)
                        else "nominal-navbar force-expand"
                    ),

                )
            )

        @reactive.calc
        def Review():
            frame, note = EligibleFrame(), AllCandidates()[1]
            panel = input.MethodPanel() or 'overall'
            if not frame.empty and panel.startswith('method:'):
                frame = frame.loc[frame.Model == panel[7:]].reset_index(drop=True)
            if input.UsuallyTransparentOnly() and not AllCandidates()[0].empty:
                excluded = len(AllCandidates()[0]) - len(EligibleFrame())
                note = (note + ' ' if note else '') + f'Intrinsic transparency required: {excluded} trials excluded; only Usually transparent methods are eligible.'
                if EligibleFrame().empty:
                    note += ' No completed trials have a verified Usually transparent method classification.'
            return frame, note

        @output(suspend_when_hidden=False)
        @render.data_frame
        def Candidates():
            nonlocal restoring_selection
            with reactive.isolate(): restoring_selection = selected_trial() is not None
            frame = Review()[0].copy()
            if not frame.empty:
                decision = Decision()[0]
                frame.insert(2, 'Overall winner', frame.Trial == decision['overall'])
                frame.insert(3, 'Method winner', frame.apply(lambda row: decision['methods'].get(str(row.Model)) == row.Trial, axis=1))
            styles = [] if frame.empty else [{
                'rows': [i for i, value in enumerate(frame.Assessment) if value == 'Best score (reference)'],
                'class': 'model-candidates-reference'}]
            return render.DataGrid(frame.round(4), selection_mode='row', height='350px', filters=False, styles=styles)

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
            nonlocal camera
            widget = plot_instance()
            req(widget is not None)
            results, frame = Results(), Review()[0]
            decision = Decision()[0]
            if results is None or frame.empty:
                figure = Card.empty_figure(AllCandidates()[1] if AllCandidates()[0].empty else Review()[1] or 'No eligible configurations in this view.')
            else:
                metric = results['state']['specification']['metric']
                panel = input.MethodPanel() or 'overall'
                winner = decision['overall'] if panel == 'overall' else decision['methods'].get(panel[7:])
                colours = {key: i for i, key in enumerate(AllCandidates()[0].Model.unique())}
                title = 'Overall' if panel == 'overall' else str(frame.iloc[0].Method)
                figure = pareto_figure(frame, metric, winner, title, colours)
                figure.update_layout(modebar={'orientation': 'v'})
                if not this.isFullScreen():
                    figure.update_layout(
                        title_text='', 
                        showlegend=False, 
                        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
                        font={'size': 13}
                    )
            current_camera = widget.layout.scene.camera.to_plotly_json()
            if current_camera:
                camera = current_camera
            if camera and any(trace.type == 'scatter3d' for trace in figure.data):
                figure.update_layout(scene_camera=camera)
            config = {'displayModeBar': bool(this.isFullScreen()), 'displaylogo': False, 'responsive': True}
            if update_figure(widget, figure, config):
                await session.send_custom_message('update_plotly_config', {'id': this.ns('ReviewPlot'), 'config': config})
            def InspectPoint(trace, points, selector):
                if points.point_inds and trace.meta and 'trials' in trace.meta:
                    selected_trial.set(int(trace.meta['trials'][points.point_inds[0]]))
            for trace in widget.data:
                if trace.meta:
                    trace.on_click(InspectPoint)

        @output
        @render.text
        def WeightSummary():
            weights, _, error = Weighting()
            if error:
                return error
            label = 'Error rate' if Results() and Results()['state']['specification']['task'] == 'classification' else 'P95 error'
            return f'Score {weights[0]:.0%} · Throughput {weights[1]:.0%} · {label} {weights[2]:.0%}'

        @output
        @render.text
        def Note():
            return Review()[1]

        @output
        @render.text
        def Status():
            frame = Review()[0]
            if frame.empty:
                return AllCandidates()[1] if AllCandidates()[0].empty else Review()[1] or 'No eligible configurations in this view.'
            decision, error = Decision()
            if error:
                return error
            panel = input.MethodPanel() or 'overall'
            winner = decision['overall'] if panel == 'overall' else decision['methods'].get(panel[7:])
            selected = frame.loc[frame.Trial == winner]
            name = f'Trial {winner} · {selected.iloc[0].Method}' if not selected.empty else 'No eligible weighted winner; check required evidence.'
            suffix = ' · Historical review (read-only)' if Results().get('read_only') else ''
            return f'{len(frame)} trials · {frame.Model.nunique()} methods · {name}{suffix}'

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

        @output
        @render.download_button(filename='candidate-models.csv')
        def ResultsCSV():
            frame = Review()[0].copy()
            decision = Decision()[0]
            if not frame.empty:
                frame['Overall winner'] = frame.Trial == decision['overall']
                frame['Method winner'] = frame.apply(lambda row: decision['methods'].get(str(row.Model)) == row.Trial, axis=1)
            yield frame.to_csv(index=False)

        @this.reactable(calc=True)
        def Exported():
            source = Incoming()
            req(source is results_source())
            results = Results()
            decision = Decision()[0]
            if results is None:
                return source
            weights, floor, _ = Weighting()
            review = {'version': 2, 'policy': 'weighted_product',
                      'usually_transparent_only': bool(input.UsuallyTransparentOnly()),
                      'weights': weights.tolist() if weights is not None else None, 'error_floor': floor,
                      'winner_trial': decision['overall'] - 1 if decision['overall'] is not None else None,
                      'accepted_trial': decision['overall'] - 1 if decision['overall'] is not None and not results.get('read_only') else None,
                      'representatives': {key: value - 1 for key, value in decision['methods'].items() if value is not None}}
            source = source.with_tuning_results({**results, 'review': review})
            if decision['overall'] is not None and not results.get('read_only'):
                return source.with_selected_model(decision['overall'] - 1, card='cand_models', operation='Seek optimal model')
            return source.with_inactive_step(stage='Learning', card='cand_models', operation='Select optimal model(s)')

        return Exported

    this.server = server
    return this

if Module.running_directly(name=__name__):
    from evaluation_plan import build_plan
    from model_commit import prepare_predictors
    from model_review import completed_evidence
    from pipeline_search import fresh, prepare, run
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
    this._imports.set(source)
    this.run()
