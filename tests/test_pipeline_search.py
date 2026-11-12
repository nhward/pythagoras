"""Search correctness: fold isolation, continuation and fixed acceptance."""
import copy
import json
import threading
import time
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from evaluation_plan import build_plan, level_key
from model_commit import prepare_predictors
from pipeline_search import (
    SearchWorker,
    accepted_parameters,
    fresh,
    mean_score,
    prepare,
    run,
    validate_checkpoint,
)
from proxy_data import proxy_data
from roles import RoleMap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from tuning import discover, fingerprint

pytestmark = pytest.mark.unit


def source():
    x = np.arange(24, dtype=float)
    frame = pd.DataFrame({'x': x, 'y': 2 * x + np.sin(x), 'partition': ['dev'] * 20 + ['held'] * 4})
    data = proxy_data(frame, _roles=RoleMap.from_primitive({
        'predictor': ['x'], 'target': ['y'], 'partition': ['partition']}))
    plan = build_plan(data, strategy='kfold', objective='independent', folds=3,
        partition_map={level_key('dev'): 'development', level_key('held'): 'holdout'})
    assert plan.ready, plan.errors
    data = data.with_evaluation_plan(plan, card='obs_resample')
    model = Pipeline([('predictors', FunctionTransformer(prepare_predictors,
        kw_args={'columns': ['x'], 'preparation': 'numeric'})),
        ('scale', StandardScaler()), ('estimator', Ridge(alpha=1.))])
    return data.with_estimator(model, label='Ridge', recipe_id='test-ridge')


def problem(**kwargs):
    return prepare(source(), model_values={'data_models__estimator__alpha': [.01, .1, 10.]},
                   metric='neg_root_mean_squared_error', **kwargs)


def finish(p, checkpoint=None, trials=20, publish=lambda s: None, pause=lambda: False):
    return run(p, checkpoint or fresh(p), trials=trials, seconds=0, publish=publish, pause=pause)


def test_grid_baseline_exhaustion_holdout_and_commit():
    data = source()
    p = prepare(data, model_values={'data_models__estimator__alpha': [.01, 10.]}, method='grid')
    holdout = set(data.evaluation_plan.holdout)
    assert holdout == set(range(20, 24))
    assert all(not ((set(a) | set(b)) & holdout) for a, b in p.splits)
    before = data.pipeline.get_params()['data_models__estimator__alpha']
    result = finish(p)
    assert result['status'] == 'exhausted'
    assert len(result['trials']) == 3
    assert all(t['state'] == 'complete' and len(t['folds']) == 3 for t in result['trials'])
    best = max(result['trials'], key=lambda t: mean_score(t, p.specification['metric']))
    parameters = accepted_parameters(p, result, best['id'])
    tuned = data.with_tuned_parameters(parameters, fingerprint=p.specification['fingerprint'],
        trial=best['id'], metric=p.specification['metric'], score=mean_score(best, p.specification['metric']))
    assert data.pipeline.get_params()['data_models__estimator__alpha'] == before
    assert tuned.pipeline.get_params()['data_models__estimator__alpha'] == best['params']['data_models__estimator__alpha']
    assert not hasattr(tuned.pipeline.named_steps.data_models.named_steps.estimator, 'coef_')
    pd.testing.assert_frame_equal(data.clean_frame, tuned.clean_frame)
    assert tuned.evaluation_plan.holdout_used is False
    assert tuned.processing_records[-1].card == 'pipeline_tune'


def test_fold_pause_json_resume_matches_uninterrupted():
    p = problem(method='random')
    stop = threading.Event()
    def publish(state):
        if state['trials'] and len(state['trials'][0]['folds']) == 1: stop.set()
    paused = finish(p, publish=publish, pause=stop.is_set)
    assert paused['status'] == 'paused'
    assert len(paused['trials'][0]['folds']) == 1
    with pytest.raises(ValueError, match='completed candidate'): accepted_parameters(p, paused, 0)
    resumed = finish(p, json.loads(json.dumps(paused)))
    complete = finish(p)
    assert [(t['params'], t['folds']) for t in resumed['trials']] == [(t['params'], t['folds']) for t in complete['trials']]
    assert len({json.dumps(t['params'], sort_keys=True) for t in resumed['trials']}) == 4


def test_fingerprint_rejects_changed_values_space_splits_and_incomplete_scores():
    p = problem()
    state = finish(p)
    changed = problem(seed=7)
    with pytest.raises(ValueError, match='does not match'): validate_checkpoint(changed, state)
    data = source(); data.clean_frame.loc[0, 'x'] = 1234
    with pytest.raises(ValueError): prepare(data)
    bad = copy.deepcopy(state); bad['trials'][0]['folds'].pop()
    with pytest.raises(ValueError, match='every evaluation fold'): validate_checkpoint(p, bad)


def test_failures_secondary_metrics_and_budget_extension():
    p = prepare(source(), model_values={'data_models__estimator__alpha': [-1., .1]},
        method='grid', secondary=['neg_mean_absolute_error'])
    initial = finish(p, trials=1)
    assert initial['status'] == 'budget reached'
    assert initial['trials'][1]['state'] == 'failed'
    assert initial['trials'][1]['error']
    final = finish(p, initial, trials=2)
    assert final['trials'][2]['state'] == 'complete'
    assert 'neg_mean_absolute_error' in final['trials'][2]['folds'][0]


def test_preprocessing_and_model_parameters_form_one_space():
    data = source().with_pipeline_step(
        StandardScaler().set_output(transform='pandas'), name='scale', preview_frame=source().frame)
    spec = {'fingerprint': fingerprint(discover(data.pipeline)),
            'grid': {'scale__with_mean': [False, True]}, 'combinations': 2}
    data = data.with_tuning_spec(spec)
    p = prepare(data, model_values={'data_models__estimator__alpha': [.1, 10.]}, method='grid')
    assert p.specification['combinations'] == 4
    state = finish(p)
    assert len(state['trials']) == 5
    assert data.tuning_spec == spec


def test_tpe_resume_uses_truthful_finite_history():
    pytest.importorskip('optuna')
    p = problem(method='tpe')
    initial = finish(p, trials=1)
    result = finish(p, json.loads(json.dumps(initial)))
    assert len(result['trials']) == 4
    assert all(t['state'] == 'complete' for t in result['trials'])


class FoldAudit(TransformerMixin, BaseEstimator):
    fitted_rows: ClassVar[list[tuple[tuple[int, ...], float]]] = []
    validation_rows: ClassVar[list[tuple[int, ...]]] = []

    def fit(self, X, y=None):
        type(self).fitted_rows.append((tuple(X.index), float(X.x.mean())))
        return self

    def transform_training_x_y(self, X, y):
        return X.iloc[1:], y.iloc[1:]

    def transform_x_y(self, X, y):
        return X, y

    def transform(self, X):
        type(self).validation_rows.append(tuple(X.index))
        return X


def test_fold_fitting_uses_raw_training_rows_and_preserves_validation_rows():
    FoldAudit.fitted_rows.clear(); FoldAudit.validation_rows.clear()
    data = source()
    data = data.with_pipeline_step(FoldAudit(), name='audit', preview_frame=data.frame.assign(x=-9999.))
    p = prepare(data, method='grid')
    result = finish(p)
    assert all(t['state'] == 'complete' for t in result['trials'])
    assert FoldAudit.fitted_rows == [(tuple(a), float(data.clean_frame.x.iloc[a].mean())) for a, _ in p.splits]
    assert FoldAudit.validation_rows == [tuple(b) for _, b in p.splits
        for _ in [p.specification['metric'], *p.specification['secondary']]]


@pytest.mark.parametrize('alter', [lambda s: s.pop('rng'), lambda s: s.update(trials=[{}]),
    lambda s: s.update(rng={'invalid': True}), lambda s: s.update(trials='bad')])
def test_malformed_checkpoint_is_a_validation_error(alter):
    p = problem(); state = fresh(p); alter(state)
    with pytest.raises(ValueError): validate_checkpoint(p, state)


def test_weighted_and_unsupported_task_guard():
    data = source()
    data.role_map.set_roles('x', ['weighting'])
    # Changed roles may invalidate evaluation first; both conditions must block.
    with pytest.raises(ValueError): prepare(data)


class SlowRidge(Ridge):
    def fit(self, X, y, **kwargs):
        time.sleep(5)
        return super().fit(X, y, **kwargs)


def test_isolated_worker_can_interrupt_fit_and_resume_checkpoint(monkeypatch):
    data = source()
    model = data.pipeline.named_steps.data_models
    model.set_params(estimator=SlowRidge())
    data = data.with_estimator(model, label='Slow Ridge', recipe_id='slow-ridge')
    p = prepare(data, method='grid')
    worker = SearchWorker()
    checkpoint = fresh(p)
    try:
        worker.start(p, checkpoint, trials=1, seconds=0)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not checkpoint['trials']:
            checkpoint = worker.poll() or checkpoint
            time.sleep(.03)
        assert checkpoint['trials'] and checkpoint['trials'][0]['state'] == 'running'
        tick = time.monotonic()
        worker.close()
        assert time.monotonic() - tick < 4
        assert not worker.running
        assert not checkpoint['trials'][0]['folds']
        monkeypatch.setattr(SlowRidge, 'fit', Ridge.fit)
        completed = finish(p, checkpoint)
        assert completed['trials'][0]['state'] == 'complete'
    finally:
        worker.close()


def test_classification_and_positive_displayed_loss():
    from pipeline_search import displayed
    from sklearn.linear_model import LogisticRegression
    data = source()
    frame = data.clean_frame.copy()
    frame['y'] = pd.Categorical(np.where(np.arange(len(frame)) % 2, 'yes', 'no'))
    data = proxy_data(frame, _roles=data.role_map)
    plan = build_plan(data, strategy='kfold', objective='independent', folds=3,
        partition_map={level_key('dev'): 'development', level_key('held'): 'holdout'})
    data = data.with_evaluation_plan(plan, card='obs_resample')
    model = Pipeline([('predictors', FunctionTransformer(prepare_predictors,
        kw_args={'columns': ['x'], 'preparation': 'numeric'})), ('estimator', LogisticRegression())])
    data = data.with_estimator(model, label='Logistic', recipe_id='logistic')
    p = prepare(data, metric='neg_log_loss', secondary=['balanced_accuracy'])
    result = finish(p)
    baseline = result['trials'][0]
    assert baseline['state'] == 'complete'
    assert displayed(mean_score(baseline, 'neg_log_loss'), 'neg_log_loss') > 0
    assert 0 <= mean_score(baseline, 'balanced_accuracy') <= 1


def test_all_metrics_survive_json_resume():
    from pipeline_search import METRICS
    p = problem(method='grid')
    result = finish(p)
    assert set(p.specification['secondary']) == set(METRICS['regression']) - {p.specification['metric']}
    fold = result['trials'][0]['folds'][0]
    assert set(fold) == {p.specification['metric'], *p.specification['secondary'], 'metric_errors'}
    validate_checkpoint(p, json.loads(json.dumps(result)))
    bad = copy.deepcopy(result); bad['version'] = 1
    with pytest.raises(ValueError, match='does not match'): validate_checkpoint(p, bad)


def test_secondary_failure_does_not_fail_trial(monkeypatch):
    import pipeline_search
    original = pipeline_search.get_scorer
    def scorer(metric):
        if metric == 'r2':
            def fail(*args): raise ValueError('metric unavailable on this fold')
            return fail
        return original(metric)
    monkeypatch.setattr(pipeline_search, 'get_scorer', scorer)
    p = prepare(source(), method='grid')
    result = finish(p)
    assert result['trials'][0]['state'] == 'complete'
    assert mean_score(result['trials'][0], 'r2') is None
    assert 'metric unavailable' in result['trials'][0]['folds'][0]['metric_errors']['r2']
    validate_checkpoint(p, result)
    from cards.pipeline_tune import trial_frame
    table = trial_frame(result)
    assert 'Complexity' not in table.columns
    assert 'r2:' in table['Metric issues'].iloc[0]


def test_exhaustive_unlimited_and_baseline_overlap_coverage():
    from pipeline_search import grid_coverage
    p = prepare(source(), model_values={'data_models__estimator__alpha': [1., 10.]}, method='grid')
    partial = run(p, fresh(p), trials=1, publish=lambda s: None, pause=lambda: False)
    # Baseline already covers one grid configuration; the cap must not hide exhaustion.
    assert partial['status'] == 'exhausted'
    assert grid_coverage(partial) == {'complete': 2, 'failed': 0, 'visited': 2, 'total': 2}
    p = problem(method='grid')
    partial = finish(p, trials=1)
    assert partial['status'] == 'budget reached'
    assert grid_coverage(partial)['complete'] == 1
    result = run(p, partial, publish=lambda s: None, pause=lambda: False)
    assert result['status'] == 'exhausted'
    assert grid_coverage(result)['complete'] == 3


def test_time_limit_preserves_partial_grid():
    from pipeline_search import grid_coverage
    p = problem(method='grid')
    state = fresh(p); state['elapsed'] = 10.
    result = run(p, state, seconds=5, publish=lambda s: None, pause=lambda: False)
    assert result['status'] == 'budget reached'
    assert grid_coverage(result)['complete'] == 0


def gp_problem():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    data = source()
    model = data.pipeline.named_steps.data_models
    model.set_params(estimator=GaussianProcessRegressor(kernel=RBF(.5), alpha=.1, optimizer=None))
    data = data.with_estimator(model, label='GP', recipe_id='test-gp')
    return prepare(data, model_values={'data_models__estimator__alpha': [.2,.3,.4]}, method='grid')








def source_without_plan(classification=False):
    from sklearn.linear_model import LogisticRegression
    data = source()
    frame = data.clean_frame[['x','y']].copy()
    if classification:
        frame['y'] = pd.Categorical(np.where(np.arange(len(frame)) % 2, 'yes', 'no'))
    raw = proxy_data(frame, _roles=RoleMap.from_primitive({'predictor':['x'], 'target':['y']}))
    model = data.pipeline.named_steps.data_models
    if classification:
        model.set_params(estimator=LogisticRegression())
    return raw.with_estimator(model, label='Test', recipe_id='test-implicit')


@pytest.mark.parametrize('method', ['random','grid','tpe','skopt'])
def test_all_search_methods_generate_reproducible_cv_when_no_plan(method):
    data = source_without_plan()
    p = prepare(data, method=method, model_values={'data_models__estimator__alpha':[.01,10.]})
    assert p.specification['evaluation']['origin'] == 'implicit'
    assert len(p.splits) == 5
    same = prepare(data, method=method, model_values={'data_models__estimator__alpha':[.01,10.]})
    assert p.specification['fingerprint'] == same.specification['fingerprint']
    for (train, valid), (train2, valid2) in zip(p.splits,same.splits):
        np.testing.assert_array_equal(train,train2)
        np.testing.assert_array_equal(valid,valid2)
        assert not set(train) & set(valid)
    state = finish(p, trials=1)
    resumed = finish(p,json.loads(json.dumps(state)))
    complete = finish(p)
    assert [(t['params'],t['folds']) for t in resumed['trials']] == [(t['params'],t['folds']) for t in complete['trials']]
    assert all(t['state']=='complete' for t in complete['trials'])
    assert data.evaluation_plan is None


def test_implicit_classification_stratifies_and_excludes_unlabelled_rows():
    data = source_without_plan(classification=True)
    data.clean_frame.loc[[22,23],'y'] = np.nan
    p = prepare(data)
    assert p.specification['evaluation']['strategy'] == 'stratified'
    assert p.specification['evaluation']['prediction'] == [22,23]
    for train,valid in p.splits:
        assert not ({22,23} & (set(train)|set(valid)))
        assert set(data.clean_frame.y.iloc[train]) == {'yes','no'}
        assert set(data.clean_frame.y.iloc[valid]) == {'yes','no'}
    assert finish(p)['trials'][0]['state'] == 'complete'


def test_implicit_cv_reduces_small_fold_counts_and_rejects_ambiguous_roles():
    from dataclasses import replace
    data = source_without_plan()
    small = proxy_data(data.clean_frame.iloc[:3].copy(), _roles=data.role_map)
    small = small.with_estimator(data.pipeline.named_steps.data_models, label='Ridge', recipe_id='test')
    assert len(prepare(small).splits) == 3
    for role in ('partition','sequence','geometry','stratifier'):
        raw = source_without_plan()
        raw.role_map.set_roles('x',[role])
        with pytest.raises(ValueError, match='upstream evaluation plan'):
            prepare(raw)
    # A stale or invalid upstream plan must never trigger the fallback.
    raw = source()
    raw = replace(raw, _evaluation_plan=raw.evaluation_plan.invalidate('Invalid plan for test'))
    with pytest.raises(ValueError, match='Invalid plan for test'):
        prepare(raw)


def test_skopt_uses_exact_upstream_memberships_and_real_finite_history(monkeypatch):
    import pipeline_search
    import skopt
    p = problem(method='skopt')
    data = source()
    for actual,expected in zip(p.splits, data.evaluation_plan.splits(data)):
        for a,b in zip(actual,expected): np.testing.assert_array_equal(a,b)
    state = finish(p)
    told=[]
    class OptimizerSpy:
        def __init__(self,*args,**kwargs): pass
        def tell(self,points,losses): told.extend(zip(points,losses))
        def ask(self): return [0]
    monkeypatch.setattr(skopt,'Optimizer',OptimizerSpy)
    pipeline_search.skopt_proposal(p,state)
    assert len(told) == 3  # The out-of-grid baseline is not given invented coordinates.
    assert [loss for _,loss in told] == [-mean_score(t,p.specification['metric']) for t in state['trials'][1:]]
    state['trials'][1]['state'] = 'failed'
    told.clear()
    pipeline_search.skopt_proposal(p,state)
    assert len(told)==2  # No penalty score is fabricated for a failed fit.


def test_skopt_fits_actual_surrogate_after_initial_observations():
    from pipeline_search import skopt_proposal
    p = prepare(source_without_plan(), method='skopt',
                model_values={'data_models__estimator__alpha':[float(i) for i in range(1,13)]})
    state = fresh(p)
    # Exercise the fitted GP acquisition path, beyond skopt's 10 initial points.
    for i in range(10):
        state['trials'].append({'params':{'data_models__estimator__alpha':float(i+1)},
            'state':'complete','folds':[{p.specification['metric']:-float(i+1)}]})
    assert skopt_proposal(p,state)['data_models__estimator__alpha'] in p.specification['grid']['data_models__estimator__alpha']


@pytest.mark.parametrize('method', ['random','grid','tpe','skopt'])
def test_zero_candidate_budget_means_unlimited_for_every_method(method):
    p = problem(method=method)
    limited = finish(p,trials=1)
    assert limited['status'] == 'budget reached'
    unlimited = finish(p,json.loads(json.dumps(limited)),trials=0)
    assert unlimited['status'] == 'exhausted'
    assert len(unlimited['trials']) == 4
    assert all(t['state']=='complete' for t in unlimited['trials'])


def test_gp_search_publishes_chart_only_at_trial_completion(monkeypatch):
    import pipeline_search
    from cards.pipeline_tune import chart_snapshot
    # p = gp_problem()
    monkeypatch.setattr(pipeline_search,'get_scorer',lambda metric: lambda *args: -1.)
    last = None
    plotted_counts=[]
    def publish(state):
        nonlocal last
        snapshot = chart_snapshot(state)
        if snapshot != last:
            plotted_counts.append(len(snapshot['trials']))
            last = snapshot
    # state = finish(p,trials=0,publish=publish)
    assert plotted_counts == [0,1,2,3,4]




def test_catboost_search_can_tune_previously_unset_parameter():
    from catboost import CatBoostRegressor
    model = Pipeline([('predictors', FunctionTransformer(prepare_predictors,
        kw_args={'columns': ['x'], 'preparation': 'numeric'})),
        ('estimator', CatBoostRegressor(iterations=3, verbose=False, thread_count=1))])
    data = source().with_estimator(model, label='CatBoost', recipe_id='test-catboost')
    key = 'data_models__estimator__depth'
    p = prepare(data, model_values={key: [2]}, method='grid')
    result = finish(p, trials=0)
    assert result['status'] == 'exhausted'
    assert [t['params'][key] for t in result['trials']] == [None, 2]
    assert all(t['state'] == 'complete' for t in result['trials'])
    validate_checkpoint(p, json.loads(json.dumps(result)))


def test_gp_search_fits_each_trial_fold_once_and_records_scores_only(monkeypatch):
    from sklearn.gaussian_process import GaussianProcessRegressor
    p = gp_problem()
    fits = []
    original = GaussianProcessRegressor.fit
    def fit(self, X, y, **kwargs):
        fits.append(len(X))
        return original(self, X, y, **kwargs)
    monkeypatch.setattr(GaussianProcessRegressor, 'fit', fit)
    result = finish(p, trials=0)
    assert result['status'] == 'exhausted'
    assert len(fits) == len(result['trials']) * len(p.splits)
    metrics = {p.specification['metric'], *p.specification['secondary'], 'metric_errors'}
    assert all(set(fold) == metrics for t in result['trials'] for fold in t['folds'])
    assert 'complexity' not in json.dumps(result).lower()
    old = copy.deepcopy(result)
    old['version'] = 3
    with pytest.raises(ValueError, match='does not match'):
        validate_checkpoint(p, old)


def test_repeated_upstream_plan_keeps_split_identity_through_resume():
    data = source()
    repeated = build_plan(data, strategy='kfold', objective='independent', folds=3, repeats=2,
        partition_map={level_key('dev'): 'development', level_key('held'): 'holdout'})
    data = data.with_evaluation_plan(repeated, card='obs_resample')
    p = prepare(data, method='grid')
    assert len(p.splits) == 6
    assert p.specification['evaluation']['split_descriptors'] == list(repeated.split_descriptors)
    assert p.specification['evaluation']['repeats'] == 2
    stopped = False
    def publish(state):
        nonlocal stopped
        stopped = bool(state['trials'] and len(state['trials'][0]['folds']) >= 1)
    checkpoint = finish(p, publish=publish, pause=lambda: stopped)
    assert len(checkpoint['trials'][0]['folds']) == 1
    checkpoint = finish(p, checkpoint=json.loads(json.dumps(checkpoint)))
    restored = validate_checkpoint(p, json.loads(json.dumps(checkpoint)))
    assert restored['specification']['evaluation']['split_descriptors'] == list(repeated.split_descriptors)
    assert len(restored['trials'][0]['folds']) == 6
