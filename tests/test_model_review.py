"""Evidence transfer, honest equivalence, regression/classification trade-offs and acceptance."""
import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from evaluation_plan import build_plan, level_key
from model_review import (
    completed_evidence,
    diagnostics,
    error_figure,
    parameter_differences,
    parameter_label,
    pareto_mask,
    representatives,
    review_frame,
    tradeoff_figure,
    within_figure,
)
from pipeline_search import fresh, prepare, run, validate_checkpoint
from test_multi_model_search import configured

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(('path', 'expected'), [
    ('cand_methods__estimator__depth', 'Cat Boost: estimator-depth'),
    ('cand_methods__estimator_depth', 'Cat Boost: estimator-depth'),
    ('cand_methods__estimator__learning_rate', 'Cat Boost: estimator-learning-rate'),
    ('var_transform__transformer__method', 'Var transform: transformer-method'),
])
def test_parameter_labels_use_method_context_without_exposing_pipeline_separators(path, expected):
    assert parameter_label(path, {'model_id': 'catboost', 'method': 'Cat Boost Regressor'}) == expected


@pytest.fixture(scope='module')
def evaluated():
    source = configured()
    source = source.with_evaluation_plan(build_plan(source, strategy='kfold', objective='independent', folds=3,
                                                   repeats=2, partition_map={level_key('dev'): 'development', level_key('held'): 'holdout'}), card='obs_resample')
    problem = prepare(source, method='grid')
    state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    assert all(t['state'] == 'complete' for t in state['trials'])
    return source, problem, state


def test_evidence_has_bounded_summaries_unfitted_templates_and_no_holdout(evaluated):
    source, problem, state = evaluated
    results = completed_evidence(problem, state)
    validate_checkpoint(problem, json.loads(json.dumps(state)))
    for trial in state['trials']:
        for fold_id, fold in zip(trial['fold_ids'], trial['folds']):
            evidence = fold['evidence']
            assert 'error' not in evidence
            assert evidence['count'] == len(problem.splits[fold_id][1])
            assert not set(problem.splits[fold_id][1]) & set(source.evaluation_plan.holdout)
            assert not {'positions', 'observed', 'predicted', 'probabilities', 'key'} & evidence.keys()
        assert len(trial['diagnostics']['exceedance']) == 101
    assert len(results['pipelines'].templates) == 2
    for pipeline in results['pipelines'].values():
        model = pipeline.steps[-1][1].steps[-1][1]
        assert not hasattr(model, 'coef_') and not hasattr(model, 'tree_')
    assert results['state']['specification']['evaluation']['repeats'] == 2


def test_proxy_evidence_is_defensive_survives_clone_acceptance_and_invalidates_on_changes(evaluated):
    source, problem, state = evaluated
    data = source.with_tuning_results(completed_evidence(problem, state))
    original = data.tuning_results
    changed = data.tuning_results
    changed['state']['trials'].clear()
    assert data.tuning_results['state']['trials']
    assert data.clone().tuning_results is not None
    trial = state['trials'][-1]
    selected = data.with_selected_model(trial['id'])
    assert selected.tuning_results is not None
    assert selected.selected_model_candidate == trial['params']['__method__']
    assert selected.pipeline_for_training().get_params()['cand_methods__estimator'].__class__ == original['pipelines'][str(trial['id'])].get_params()['cand_methods__estimator'].__class__
    assert not data.equals(source)
    frame = source.clean_frame.copy()
    frame.loc[0, 'x'] += 1
    assert replace(data, _df=frame, _clean_df=frame).tuning_results is None
    assert data.with_estimator(source.model_candidates[0]['estimator'], label='Other', recipe_id='other').tuning_results is None


def test_per_method_shortlists_and_no_false_equivalence_without_tolerance(evaluated):
    _, problem, state = evaluated
    results = completed_evidence(problem, state)
    frame, note = review_frame(results, limit=2)
    assert frame.groupby('Model').size().tolist() == [2, 2]
    assert (frame.Assessment == 'Best score (reference)').sum() == 2
    assert frame['P(equivalent)'].isna().all()
    assert 'tolerance' in note
    assert frame['P95 absolute error'].notna().all()
    assert (frame['Observed rows/s'] > 0).all()
    assessed, _ = review_frame(results, tolerance=1000, probability=.9)
    assert 'Practically equivalent' in assessed.Assessment.values
    assert not any('complexity' in c.lower() for c in frame.columns)


def test_diagnostic_percentiles_are_calculated_per_repeat(evaluated):
    _, problem, state = evaluated
    trial = state['trials'][0]
    expected = []
    for repeat in (1, 2):
        residuals = []
        from sklearn.base import clone
        for fold_id in trial['fold_ids']:
            if problem.specification['evaluation']['split_descriptors'][fold_id]['repeat'] == repeat:
                train, validation = problem.splits[fold_id]
                child = problem.branches[trial['params']['__method__']]
                parameters = {k: v for k, v in trial['params'].items() if k != '__method__'}
                pipeline = clone(child.pipeline).set_params(**parameters)
                pipeline.fit(problem.frame.iloc[train], problem.frame[problem.target].iloc[train])
                predicted = pipeline.predict(problem.frame.iloc[validation])
                residuals.extend(np.abs(problem.frame[problem.target].iloc[validation].to_numpy() - predicted))
        expected.append(np.quantile(residuals, .95))
    assert diagnostics(trial, problem.specification)['P95 absolute error'] == pytest.approx(np.mean(expected))


def test_pareto_directions_missing_evidence_representatives_and_plots(evaluated):
    _, problem, state = evaluated
    results = completed_evidence(problem, state)
    frame, _ = review_frame(results, tolerance=.2)
    reps = representatives(frame, {})
    assert len(reps) == 2
    assert len(tradeoff_figure(reps, ('Score', 'P95 absolute error'), problem.specification['metric']).data) == 2
    assert len(error_figure(results, reps).data) == 2
    assert len(within_figure(frame, frame.iloc[0].Model, .2).data) == 1
    changed = representatives(frame, {frame.iloc[0].Model: int(frame.iloc[1].Trial)})
    assert changed.iloc[0].Trial == frame.iloc[1].Trial
    assert not parameter_differences(results, int(frame.iloc[1].Trial), int(frame.iloc[0].Trial)).empty
    test = pd.DataFrame({'Score': [1., 2., 1.5, np.nan], 'Observed rows/s': [10., 5., 4., 100.]})
    assert pareto_mask(test, ('Score', 'Observed rows/s'), 'neg_mean_absolute_error').tolist() == [True, False, False, False]
    assert pareto_mask(test, ('Score', 'Observed rows/s'), 'accuracy').tolist() == [True, True, False, False]


def test_classification_uses_probabilities_error_rate_and_log_loss():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from test_pipeline_search import source
    data = source()
    frame = data.clean_frame.copy()
    frame['y'] = pd.Categorical(np.where(frame.x > 10, 'yes', 'no'))
    data = replace(data, _df=frame, _clean_df=frame)
    data = data.with_estimator(Pipeline([('estimator', LogisticRegression())]), label='Logistic', recipe_id='logistic')
    # Explicit predictor selection prevents the Target from reaching the model.
    from model_commit import prepare_predictors
    from sklearn.preprocessing import FunctionTransformer
    model = Pipeline([('predictors', FunctionTransformer(prepare_predictors, kw_args={'columns': ['x'], 'preparation': 'numeric'})),
                      ('estimator', LogisticRegression())])
    data = data.with_estimator(model, label='Logistic', recipe_id='logistic')
    data = data.with_evaluation_plan(build_plan(data, strategy='kfold', objective='independent', folds=3, repeats=2, partition_map={level_key('dev'): 'development', level_key('held'): 'holdout'}), card='obs_resample')
    problem = prepare(data)
    state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    results = completed_evidence(problem, state)
    assert len(results['state']['trials']) == 1
    frame, _ = review_frame(results)
    assert 'P95 absolute error' not in frame
    assert 0 <= frame.iloc[0]['Error rate'] <= 1
    assert frame.iloc[0]['Log loss'] >= 0
    assert state['trials'][0]['diagnostics']['loss_kind'] == 'log_loss'
    assert 'probabilities' not in json.dumps(state)


def test_older_evidence_is_reported_as_missing_not_invented(evaluated):
    _, problem, state = evaluated
    results = completed_evidence(problem, state)
    for trial in results['state']['trials']:
        trial.pop('diagnostics', None)
        for fold in trial['folds']:
            fold.pop('evidence', None)
    frame, _ = review_frame(results)
    assert frame['P95 absolute error'].isna().all()
    assert frame['Observed rows/s'].isna().all()
    assert frame.Evidence.str.contains('unavailable').all()


def test_checkpoint_rejects_misaligned_validation_counts(evaluated):
    _, problem, state = evaluated
    tampered = copy.deepcopy(state)
    tampered['trials'][0]['folds'][0]['evidence']['count'] += 1
    with pytest.raises(ValueError, match='aligned'):
        validate_checkpoint(problem, tampered)


def test_single_method_review_acceptance_preserves_explicit_candidate_training():
    from test_multi_model_search import candidates
    source = candidates()
    chosen = source.model_candidates[0]
    source = source.with_model_candidates([chosen])
    problem = prepare(source, method='grid')
    state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    data = source.with_tuning_results(completed_evidence(problem, state)).with_selected_model(0)
    assert data.selected_model_candidate == chosen['id']
    assert data.pipeline_for_training(chosen['id']).get_params()['cand_methods__estimator__alpha'] == 1.
    assert data.tuning_results is not None
