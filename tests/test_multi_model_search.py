"""Conditional method alternatives preserve fold fitting and checkpoint identity."""
import copy
import json

import pytest
from model_commit import prepare_predictors
from pipeline_search import (
    accepted_parameters,
    fresh,
    grid_coverage,
    prepare,
    run,
    validate_checkpoint,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor
from test_pipeline_search import source
from tuning import compile_spec, discover_source

pytestmark = pytest.mark.unit


def candidates():
    def model(estimator):
        return Pipeline([('predictors', FunctionTransformer(prepare_predictors,
            kw_args={'columns': ['x'], 'preparation': 'numeric'})), ('estimator', estimator)])
    return source().with_model_candidates([
        {'id': 'ridge', 'label': 'Ridge Regression', 'estimator': model(Ridge(alpha=1.))},
        {'id': 'tree', 'label': 'Decision Tree Regression', 'estimator': model(DecisionTreeRegressor(max_depth=2))}])


def configured():
    data = candidates()
    rows = discover_source(data)
    drafts = {}
    for row in rows:
        values = ([.1, 10.] if row.get('candidate_id') == 'ridge' and row['path'].endswith('__alpha')
                  else [2, 3, 4] if row.get('candidate_id') == 'tree' and row['path'].endswith('__max_depth') else None)
        if values:
            drafts[row['path']] = dict(row['definition'], values=values, active=True, owner=row['owner'])
    spec = compile_spec(rows, drafts)
    assert spec['combinations'] == 5
    return data.with_tuning_spec(spec)


def finish(problem, state=None, **kwargs):
    return run(problem, state or fresh(problem), trials=kwargs.pop('trials', 0), seconds=0,
               publish=kwargs.pop('publish', lambda s: None), pause=kwargs.pop('pause', lambda: False), **kwargs)


def test_candidates_are_defensive_and_training_requires_choice():
    data = candidates()
    with pytest.raises(ValueError, match='explicitly'):
        data.pipeline_for_training()
    altered = data.model_candidates
    altered[0]['estimator'].set_params(estimator__alpha=99)
    assert data.model_candidates[0]['estimator'].get_params()['estimator__alpha'] == 1
    assert len(data.clone().model_candidates) == 2
    assert data.pipeline_for_training('tree').named_steps.data_models.named_steps.estimator.max_depth == 2
    assert len(configured().tuning_spec['grid']) == 2
    assert {r['display_source'] for r in discover_source(data) if r.get('candidate_id') and r['parameter_path'].startswith('data_models__estimator__')} == {'Ridge Regression', 'Decision Tree Regression'}


@pytest.mark.parametrize('method', ['grid', 'random', 'tpe', 'skopt'])
def test_all_searches_cover_union_with_baselines_and_resume(method):
    data = configured()
    p = prepare(data, method=method)
    assert p.specification['combinations'] == 5
    complete = finish(p)
    assert complete['status'] == 'exhausted', complete.get('error')
    assert len(complete['trials']) == 6  # Two baselines; tree baseline is already in its grid.
    assert sum(t['kind'] == 'baseline' for t in complete['trials']) == 2
    assert grid_coverage(complete)['visited'] == 5
    assert {t['method'] for t in complete['trials']} == {'Ridge Regression', 'Decision Tree Regression'}
    for trial in complete['trials']:
        key = trial['params']['__method__']
        assert trial['model_id'] == key
        assert ('data_models__estimator__alpha' in trial['params']) == (key == 'ridge')
        assert ('data_models__estimator__max_depth' in trial['params']) == (key == 'tree')
    checkpoint = finish(p, trials=1)
    assert len(checkpoint['trials']) == 3
    resumed = finish(p, json.loads(json.dumps(checkpoint)))
    assert [t['params'] for t in resumed['trials']] == [t['params'] for t in complete['trials']]
    selected = next(t for t in resumed['trials'] if t['model_id'] == 'tree' and t['params']['data_models__estimator__max_depth'] == 3)
    accepted = data.with_tuned_parameters(accepted_parameters(p, resumed, selected['id']),
        fingerprint=p.specification['fingerprint'], trial=selected['id'], metric=p.specification['metric'], score=0,
        candidate_id='tree')
    actual = accepted.pipeline_for_training().named_steps.data_models.named_steps.estimator
    assert actual.max_depth == 3 and not hasattr(actual, 'tree_')
    assert len(accepted.model_candidates) == 2
    assert accepted.selected_model_candidate == 'tree'
    assert accepted.pipeline_for_training('tree').named_steps.data_models.named_steps.estimator.max_depth == 3
    assert not prepare(accepted, method='grid').branches
    bad = copy.deepcopy(resumed)
    bad['trials'][0]['params']['__method__'] = 'unknown'
    with pytest.raises(ValueError):
        validate_checkpoint(p, bad)


def test_global_budget_excludes_all_method_baselines_and_pause_is_safe():
    p = prepare(configured(), method='grid')
    result = finish(p, trials=1)
    assert len(result['trials']) == 3
    assert sum(t['kind'] == 'candidate' for t in result['trials']) == 1
    stopped = False
    def publish(state):
        nonlocal stopped
        stopped = bool(state['trials'] and state['trials'][0]['folds'])
    paused = finish(p, publish=publish, pause=lambda: stopped)
    assert paused['status'] == 'paused'
    assert len(paused['trials'][0]['folds']) == 1
    resumed = finish(p, json.loads(json.dumps(paused)))
    assert resumed['status'] == 'exhausted'
    assert all(t['state'] == 'complete' for t in resumed['trials'])


def test_candidate_changes_invalidate_identity_and_bad_commit_is_atomic():
    data = configured()
    changed = list(data.model_candidates)
    changed[0]['estimator'].set_params(estimator__alpha=2.)
    altered = data.with_model_candidates(changed)
    assert not altered.equals(data)
    assert altered.tuning_spec is None
    before = data.clone()
    with pytest.raises((TypeError, AttributeError)):
        data.with_model_candidates([changed[0], {'id':'bad', 'label':'Bad', 'estimator': object()}])
    assert data.equals(before)
    p = prepare(data, method='grid')
    with pytest.raises(ValueError, match='does not match'):
        validate_checkpoint(prepare(altered, method='grid'), fresh(p))


@pytest.mark.skipif(__import__('joblib').cpu_count() < 2, reason='Requires two CPUs')
@pytest.mark.parametrize('execution', ['trials', 'folds'])
def test_parallel_multi_method_search_and_acceptance(execution):
    p = prepare(configured(), method='grid', execution=execution, workers=2)
    result = finish(p)
    assert result['status'] == 'exhausted', result.get('error')
    assert len(result['trials']) == 6
    assert all(len(t['folds']) == 3 and t['state'] == 'complete' for t in result['trials'])
    validate_checkpoint(p, json.loads(json.dumps(result)))


def test_shared_parameter_choices_multiply_conditional_branch_sum():
    from MissingnessTransformer import MissingnessTransformer
    data = candidates()
    data = data.with_pipeline_step(MissingnessTransformer(), name='miss_map', preview_frame=data.frame)
    rows = discover_source(data)
    drafts = {}
    for row in rows:
        if row['path'] == 'miss_map__observation_threshold':
            values = [.2, .4]
        elif row.get('candidate_id') == 'ridge' and row['path'].endswith('__alpha'):
            values = [.1, 10.]
        elif row.get('candidate_id') == 'tree' and row['path'].endswith('__max_depth'):
            values = [2, 3, 4]
        else:
            continue
        drafts[row['path']] = dict(row['definition'], mode='values', values=values,
                                  active=True, owner=row['owner'])
    spec = compile_spec(rows, drafts)
    assert spec['combinations'] == 10
    p = prepare(data.with_tuning_spec(spec), method='grid')
    assert p.specification['combinations'] == 10
    assert all(child.specification['grid']['miss_map__observation_threshold'] == [.2, .4]
               for child in p.branches.values())


def test_duplicate_labels_are_disambiguated_and_candidates_survive_preprocessing():
    data = candidates()
    choices = list(data.model_candidates)
    for c in choices:
        c['label'] = 'Same label'
        c['method_id'] = c['id'] + '.Method'
    data = data.with_model_candidates(choices)
    assert len({c['label'] for c in data.model_candidates}) == 2
    assert data.clone().equals(data)
    with pytest.raises(ValueError, match='explicitly'):
        next(data.training_partitions())


def test_empty_method_grids_evaluate_baselines_only():
    result = finish(prepare(candidates(), method='grid'))
    assert result['status'] == 'exhausted'
    assert len(result['trials']) == 2
    assert all(t['kind'] == 'baseline' and t['state'] == 'complete' for t in result['trials'])


def test_classification_candidates_share_scoring_and_preprocessing_is_fold_local():
    import numpy as np
    import pandas as pd
    from evaluation_plan import build_plan
    from proxy_data import proxy_data
    from roles import RoleMap
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from test_pipeline_search import FoldAudit
    data = proxy_data(pd.DataFrame({'x': np.arange(30.), 'y': pd.Categorical(np.arange(30) % 2)}),
        _roles=RoleMap.from_primitive({'predictor':['x'], 'target':['y']}))
    data = data.with_evaluation_plan(build_plan(data, strategy='stratified', stratify='y',
        objective='independent', folds=3, repeats=2), card='obs_resample')
    FoldAudit.fitted_rows.clear()
    data = data.with_pipeline_step(FoldAudit(), name='audit', preview_frame=data.frame)
    def model(estimator):
        return Pipeline([('predictors', FunctionTransformer(prepare_predictors,
            kw_args={'columns':['x'], 'preparation':'numeric'})), ('estimator', estimator)])
    data = data.with_model_candidates([
        {'id':'logistic', 'label':'Logistic Regression', 'estimator':model(LogisticRegression())},
        {'id':'tree', 'label':'Classification Tree', 'estimator':model(DecisionTreeClassifier(max_depth=2))}])
    p = prepare(data, method='grid', metric='roc_auc')
    result = finish(p)
    assert result['status'] == 'exhausted', result.get('error')
    assert all(t['state'] == 'complete' and len(t['folds']) == 6 for t in result['trials'])
    assert len(FoldAudit.fitted_rows) == 12
    assert all(len(rows) == 20 for rows, _ in FoldAudit.fitted_rows)
    assert all(child.frame is p.frame and child.splits is p.splits for child in p.branches.values())
