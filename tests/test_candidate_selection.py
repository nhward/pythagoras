"""Weight semantics, full evidence selection, and deterministic regression frontiers."""
import numpy as np
import pandas as pd
import pytest
from candidate_selection import (
    candidate_frame,
    decisions,
    pareto_figure,
    pareto_frontier,
    weighted_winner,
    weights_from_handles,
)
from model_review import completed_evidence
from pipeline_search import fresh, prepare, run
from test_multi_model_search import configured

pytestmark = pytest.mark.unit
METRIC = 'neg_root_mean_squared_error'


def frame():
    return pd.DataFrame({'Trial': [1, 2, 3, 4], 'Model': ['a', 'a', 'b', 'b'],
                         'Method': ['A', 'A', 'B', 'B'], 'Score': [1., 2., 3., 4.],
                         'Observed rows/s': [10., 100., 1000., 1.], 'P95 absolute error': [5., 3., 1., 6.]})


@pytest.mark.parametrize(('handles', 'expected'), [((100, 100), [1, 0, 0]), ((0, 100), [0, 1, 0]), ((0, 0), [0, 0, 1]), ((40, 75), [.4, .35, .25])])
def test_slider_segments(handles, expected):
    np.testing.assert_allclose(weights_from_handles(handles), expected)


@pytest.mark.parametrize('handles', [(101, 101), (50, 40), (float('nan'), 50), (1,), None])
def test_invalid_handles(handles):
    with pytest.raises(ValueError):
        weights_from_handles(handles)


def test_winners_and_method_winners_use_all_trials_and_unit_changes_do_not_change_rank():
    data = frame()
    assert weighted_winner(data, METRIC, [1, 0, 0], 1e-12) == 1
    assert decisions(data, METRIC, [0, 1, 0], 1e-12) == {'overall': 3, 'methods': {'a': 2, 'b': 3}}
    assert weighted_winner(data, METRIC, [0, 0, 1], 1e-12) == 3
    weights = [.3, .4, .3]
    expected = weighted_winner(data, METRIC, weights, 1e-12)
    data['Score'] *= 1000
    data['P95 absolute error'] *= 1000
    data['Observed rows/s'] /= 60
    assert weighted_winner(data, METRIC, weights, 1e-9) == expected


def test_pareto_frontier_and_plot_keep_all_trials_and_highlight_winner():
    data = frame()
    assert pareto_frontier(data, METRIC).tolist() == [True, True, True, False]
    figure = pareto_figure(data, METRIC, 2, 'Overall', {'a': 0, 'b': 1})
    assert all(trace.type == 'scatter3d' for trace in figure.data)
    assert sum(len(trace.x) for trace in figure.data) == 5
    assert figure.data[-1].meta['trials'] == [2]
    assert figure.layout.uirevision == 'cand-models-camera'


def test_missing_active_evidence_excludes_trial_but_inactive_evidence_does_not():
    data = frame()
    data.loc[0, 'Observed rows/s'] = np.nan
    assert weighted_winner(data, METRIC, [1, 0, 0], 1e-12) == 1
    assert weighted_winner(data, METRIC, [.9, .1, 0], 1e-12) != 1
    data['Observed rows/s'] = np.nan
    assert weighted_winner(data, METRIC, [0, 1, 0], 1e-12) is None


def test_zero_error_floor_r2_transform_and_deterministic_nondominated_tie():
    data = frame()
    data.loc[:1, 'Score'] = 0
    # Equal primary score: second trial dominates the first on the other axes.
    assert weighted_winner(data, METRIC, [1, 0, 0], 1e-12) == 2
    data['Score'] = [.9, .8, -.1, -2.]
    assert weighted_winner(data, 'r2', [1, 0, 0], 1e-12) == 1
    with pytest.raises(ValueError, match='positive'):
        weighted_winner(data, METRIC, [1, 0, 0], 0)


def test_full_evidence_ignores_score_shortlist_and_preserves_full_pipeline_selection():
    source = configured()
    problem = prepare(source, method='grid')
    state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    results = completed_evidence(problem, state)
    data, _ = candidate_frame(results)
    assert len(data) == len(state['trials']) > 5
    assert 'RMSE/MAE' in data
    selection = decisions(data, problem.specification['metric'], [1, 0, 0], 1e-12)
    selected = source.with_tuning_results(results).with_selected_model(selection['overall'] - 1, card='cand_models', operation='Select weighted regression model')
    record = next(r for r in selected.processing_records if r.card == 'cand_models')
    assert record.parameters['trial'] == selection['overall'] - 1
    assert selected.tuning_results is not None


def test_transparency_constraint_uses_terminal_class_and_preserves_evidence():
    from copy import deepcopy
    from types import SimpleNamespace

    from candidate_selection import transparent_candidates
    from estimator_catalogue import load_catalogue
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeRegressor
    data = frame()
    # Human labels and IDs do not determine the class; even a shallow tree is Conditional.
    data['Model'] = ['local-ridge', 'local-ridge', 'sklearn.linear_model.Ridge', 'sklearn.linear_model.Ridge']
    data['Method'] = 'Misleading human label'
    before = data.copy(deep=True)
    results = {'pipelines': SimpleNamespace(templates={
        'local-ridge': Pipeline([('model', Ridge())]),
        'sklearn.linear_model.Ridge': Pipeline([('model', DecisionTreeRegressor(max_depth=1))]),
    })}
    catalogue = load_catalogue()
    saved = deepcopy(catalogue)
    eligible = transparent_candidates(data, results, catalogue)
    assert eligible.Trial.tolist() == [1, 2]
    assert decisions(eligible, METRIC, [0, 1, 0], 1e-12) == {
        'overall': 2, 'methods': {'local-ridge': 2}}
    pd.testing.assert_frame_equal(data, before)
    assert catalogue == saved


def test_transparency_constraint_fails_closed_and_can_use_saved_recipe_identity():
    from candidate_selection import transparent_candidates
    from estimator_catalogue import load_catalogue
    catalogue = load_catalogue()
    ridge = next(p['id'] for p in catalogue['profiles'] if p['estimator'] == 'sklearn.linear_model.Ridge')
    data = frame()
    data['Model'] = [ridge, ridge, 'unknown.Ridge', 'unknown.Ridge']
    eligible = transparent_candidates(data, {'pipelines': {}}, catalogue)
    assert eligible.Trial.tolist() == [1, 2]
    data['Model'] = 'unknown.Ridge'
    empty = transparent_candidates(data, {'pipelines': {}}, catalogue)
    assert empty.empty and list(empty.columns) == list(data.columns)
    assert decisions(empty, METRIC, [1, 0, 0], 1e-12) == {'overall': None, 'methods': {}}
    # A reviewed-looking value with an unreviewed status must not qualify.
    data['Model'] = ridge
    row = next(e for e in catalogue['estimators'] if e['id'] == 'sklearn.linear_model.Ridge')
    row['taxonomy']['transparency']['status'] = 'unreviewed'
    assert transparent_candidates(data, {'pipelines': {}}, catalogue).empty
