import pytest
from cards.pipeline_tune import chart_snapshot, progress_figure, trial_frame


def example(metric='neg_root_mean_squared_error', task='regression'):
    def trial(i, scores):
        return {'id': i, 'kind': 'baseline' if i == 0 else 'candidate', 'state': 'complete',
                    'folds': [{metric: s} for s in scores], 'elapsed': 1., 'model_id': 'test',
                    'params': {'alpha': i + 1}, 'error': ''}
    return {'specification': {'metric': metric, 'task': task, 'secondary': [], 'folds': 2, 'fingerprint': 'test'},
            'trials': [trial(2, [-.5, -4.5]), trial(0, [-3., -5.]), trial(1, [-1., -3.])]}


@pytest.mark.unit
def test_chart_shows_only_completed_mean_points():
    state = example()
    figure = progress_figure(state)
    assert len(figure.data) == 1
    assert list(figure.data[0].x) == [1, 2, 3]
    assert list(figure.data[0].y) == [4., 2., 2.5]
    assert figure.data[0].mode == 'markers'
    assert figure.data[0].error_y.array is None
    assert not figure.layout.shapes
    assert not figure.layout.showlegend
    assert list(trial_frame(state).Trial) == [1, 2, 3]


@pytest.mark.unit
def test_chart_preserves_maximizing_metric_values():
    state = example('accuracy', 'classification')
    for trial in state['trials']:
        trial['folds'] = [{'accuracy': .5 + .1 * trial['id']}, {'accuracy': .7 + .1 * trial['id']}]
    figure = progress_figure(state)
    assert figure.layout.yaxis.title.text == 'Accuracy · maximize'
    assert list(figure.data[0].y) == pytest.approx([.6, .7, .8])


@pytest.mark.unit
def test_chart_colors_are_stable_per_selected_method_and_snapshot():
    state = example()
    state['specification']['branches'] = {'ridge': {'method_label': 'Ridge'}, 'tree': {'method_label': 'Tree'}}
    for trial in state['trials']:
        key = 'tree' if trial['id'] == 1 else 'ridge'
        trial.update(model_id=key, method=key.title())
        trial['params']['__method__'] = key
    figure = progress_figure(chart_snapshot(state))
    assert [t.name for t in figure.data] == ['Ridge', 'Tree']
    assert list(figure.data[0].x) == [1, 3]
    assert list(figure.data[1].x) == [2]
    assert figure.data[0].marker.color != figure.data[1].marker.color
    assert figure.layout.showlegend
    color = figure.data[1].marker.color
    state['trials'] = [t for t in state['trials'] if t['model_id'] == 'tree']
    assert progress_figure(chart_snapshot(state)).data[0].marker.color == color


@pytest.mark.unit
def test_removed_chart_controls_are_absent_and_other_settings_have_guides():
    from cards.pipeline_tune import instance
    card = instance()
    settings = str(card.settings)
    assert 'SDMultiplier' not in settings and 'ChartView' not in settings
    for name in ('Metric', 'Method', 'Trials', 'Seconds', 'Execution', 'Workers', 'Seed', 'ModelValues'):
        guide = card._shepherd_steps[f'{card.namespace}-{name}_wrapper']
        assert guide['title'] and guide['text'] and guide['position'] == 'left'


@pytest.mark.unit
def test_chart_snapshot_ignores_fold_progress_but_publishes_completed_trial():
    from copy import deepcopy
    state = example()
    state['trials'] = sorted(state['trials'], key=lambda t: t['id'])
    current = state['trials'].pop()
    plotted = chart_snapshot(state)
    state = deepcopy(state)
    current.update(state='running', folds=[])
    state['trials'].append(current)
    assert chart_snapshot(state) == plotted
    current['folds'].append({state['specification']['metric']: -1.})
    state.update(status='running', elapsed=42.)
    assert chart_snapshot(state) == plotted
    current['folds'].append({state['specification']['metric']: -2.})
    current['state'] = 'complete'
    completed = chart_snapshot(state)
    assert completed != plotted
    assert len(completed['trials']) == 3
    state.update(status='paused', elapsed=43.)
    assert chart_snapshot(state) == completed
    assert chart_snapshot(None) is None




@pytest.mark.unit
def test_chart_snapshot_excludes_post_trial_diagnostics_and_is_detached():
    state = example()
    plotted = chart_snapshot(state)
    trial = state['trials'][0]
    trial.update(elapsed=999.)
    trial['folds'][0]['metric_errors']={'r2':'not available'}
    assert chart_snapshot(state) == plotted
    assert chart_snapshot(state) == plotted
    trial['folds'][0][state['specification']['metric']]=-100.
    assert chart_snapshot(state) != plotted


@pytest.mark.unit
def test_trial_table_has_no_complexity_columns():
    assert not any('complexity' in name.lower() for name in trial_frame(example()).columns)



@pytest.mark.unit
def test_starting_configuration_is_labeled_and_has_distinct_marker():
    state = example()
    trace = progress_figure(state).data[0]
    assert list(trace.marker.symbol) == ['diamond', 'circle', 'circle']
    assert trace.text[0].startswith('Starting configuration')
    assert trace.text[1].startswith('Candidate')
    assert progress_figure(state).layout.xaxis.title.text == 'Trial'
    assert list(trial_frame(state).Kind) == ['Starting configuration', 'Candidate', 'Candidate']
    assert state['trials'][1]['kind'] == 'baseline'
