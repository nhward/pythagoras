"""Split-fitted evaluation, complete pipeline composition and downstream contracts."""
import json
import threading
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from candidate_selection import candidate_frame, decisions
from ensemble import (
    ClassificationEnsemble,
    evaluate_ensemble,
    portable_ensemble_display,
    representatives,
    restore_ensemble_display,
)
from pipeline_search import METRICS
from ensemble_scenario import example_source
from evaluation_plan import build_plan
from sklearn.base import clone
from sklearn.exceptions import NotFittedError


@pytest.fixture(scope='module')
def source():
    return example_source(count=5)


def test_portable_display_preserves_all_metrics_without_fitted_or_row_data(source):
    result = evaluate_ensemble(source, tuple(representatives(source)[1]), 'mean')
    saved = json.loads(json.dumps(portable_ensemble_display(result), allow_nan=False))
    assert set(saved) == {'version', 'task', 'members', 'split_count', 'origin', 'input_signature', 'tables'}
    restored = restore_ensemble_display(saved)
    assert restored.input_signature == result.input_signature
    for metric in METRICS[result.task]:
        pd.testing.assert_frame_equal(restored.table(metric).reset_index(drop=True),
                                     result.table(metric).reset_index(drop=True), check_dtype=False)
        pd.testing.assert_frame_equal(restored.metric_table(metric), result.metric_table(metric), check_dtype=False)
    with pytest.raises(ValueError, match='unavailable or stale'):
        source.with_ensemble(restored)
    saved['tables'] = {}
    with pytest.raises(ValueError, match='Invalid saved'):
        restore_ensemble_display(saved)


def test_five_winners_fitted_per_split_and_unfitted_training_definition(source):
    winners = representatives(source)[1]
    assert len(winners) == 5
    result = evaluate_ensemble(source, tuple(winners), 'mean')
    assert len(result.members) == 5
    assert len(result.fitted) == 5
    for ensemble, (train, validation) in zip(result.fitted, result.splits):
        assert not set(train) & set(validation)
        X = source.clean_frame.iloc[list(validation)]
        predictions = [model.predict(X) for _, model in ensemble.estimators_]
        np.testing.assert_allclose(ensemble.predict(X), np.mean(predictions, axis=0))
        # Scalers are fitted only on this training partition, not the full source.
        expected = source.clean_frame.x.iloc[list(train)].mean()
        for _, model in ensemble.estimators_:
            assert model.named_steps['cand_methods'].named_steps['scale'].mean_[0] == pytest.approx(expected)
    with pytest.raises(NotFittedError):
        result.definition.predict(source.clean_frame)
    out = source.with_ensemble(result)
    assert out.ensemble_results is not None
    assert out.clone().ensemble_results is out.ensemble_results
    assert out.model_candidates == ()
    assert out.pipeline_steps == ('cand_methods',)
    assert len(out.pipeline_for_training().named_steps['cand_methods'].members) == 5
    with pytest.raises(NotFittedError):
        out.pipeline_for_training().predict(source.clean_frame)
    assert out.with_modelling_context(replace(out.modelling_context, task='classification')).ensemble_results is None
    with pytest.raises(ValueError, match='stale'):
        out.with_ensemble(result)
    table = result.table('neg_root_mean_squared_error')
    assert table.Score.is_monotonic_increasing
    assert result.table('r2').Score.is_monotonic_decreasing
    metrics = result.metric_table('r2')
    assert {'MAE', 'RMSE', 'R²'} <= set(metrics.columns)
    assert metrics['R²'].is_monotonic_decreasing
    assert metrics.Model.tolist() == result.table('r2').Model.tolist()
    np.testing.assert_allclose(metrics.RMSE, result.table('neg_root_mean_squared_error').set_index('Model').loc[metrics.Model, 'Score'])


def test_subset_median_empty_selection_and_cancellation(source):
    keys = tuple(representatives(source)[1])[:2]
    result = evaluate_ensemble(source, keys, 'median')
    assert len(result.members) == 2
    X = source.clean_frame
    np.testing.assert_allclose(result.fitted[0].predict(X), np.median(
        [model.predict(X) for _, model in result.fitted[0].estimators_], axis=0))
    with pytest.raises(ValueError, match='at least two'):
        evaluate_ensemble(source, (), 'mean')
    with pytest.raises(ValueError, match='at least two'):
        evaluate_ensemble(source, keys[:1], 'mean')
    stop = threading.Event(); stop.set()
    with pytest.raises(InterruptedError):
        evaluate_ensemble(source, keys, 'mean', stop)


def test_upstream_plan_is_reused(source):
    plan = build_plan(source, strategy='kfold', objective='independent', folds=3, repeats=2)
    updated = source.with_evaluation_plan(plan, card='obs_resample')
    # Re-evaluate upstream evidence because changing the plan invalidates it.
    from candidate_selection import candidate_frame, decisions
    from model_review import completed_evidence
    from pipeline_search import fresh, prepare, run
    problem = prepare(replace(updated, _selected_candidate=None))
    state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    evidence = completed_evidence(problem, state)
    frame, _ = candidate_frame(evidence)
    chosen = decisions(frame, problem.specification['metric'], [1, 0, 0], 1e-12)
    evidence['review'] = {'representatives': {key: value - 1 for key, value in chosen['methods'].items()}}
    updated = updated.with_tuning_results(evidence)
    result = evaluate_ensemble(updated, tuple(representatives(updated)[1]), 'mean')
    assert result.evaluation['origin'] == 'upstream'
    assert result.splits == tuple((tuple(a), tuple(b)) for a, b in plan.splits(updated))


@pytest.mark.parametrize('style', ['soft', 'hard'])
def test_classification_weighted_representatives_and_voting(style):
    source = example_source('classification')
    evidence, members = representatives(source)
    frame, _ = candidate_frame(evidence)
    assert 'Error rate' in frame
    chosen = decisions(frame, 'balanced_accuracy', [0, 0, 1], 1e-12)
    assert len(chosen['methods']) == 2
    result = evaluate_ensemble(source, tuple(members), style)
    assert isinstance(result.definition, ClassificationEnsemble)
    assert result.table('accuracy').Score.notna().all()
    for ensemble in result.fitted:
        assert set(ensemble.predict(source.clean_frame)) <= {0, 1}
        if style == 'soft':
            np.testing.assert_allclose(ensemble.predict_proba(source.clean_frame).sum(axis=1), 1)
        else:
            assert not hasattr(ensemble, 'predict_proba')
    metrics = result.metric_table('accuracy').set_index('Model')
    assert {'Accuracy', 'Balanced accuracy', 'ROC AUC', 'Log loss'} <= set(metrics.columns)
    if style == 'hard':
        assert np.isnan(metrics.loc['Ensemble', 'Log loss'])
        assert metrics.loc['Ensemble', 'Note']
    assert not hasattr(clone(result.definition), 'estimators_')


def test_classification_tradeoff_plot_and_incompatible_probabilities():
    from candidate_selection import pareto_figure
    from sklearn.svm import LinearSVC
    source = example_source('classification')
    evidence, _ = representatives(source)
    frame, _ = candidate_frame(evidence)
    winner = decisions(frame, 'balanced_accuracy', [1, 0, 0], 1e-12)['overall']
    figure = pareto_figure(frame, 'balanced_accuracy', winner, 'Classification', {})
    assert figure.layout.scene.zaxis.title.text == 'Error rate'
    assert figure.data
    with pytest.raises(ValueError, match='requires probabilities'):
        ClassificationEnsemble((('svm', LinearSVC()),), 'soft').fit(
            source.clean_frame[['x']], source.clean_frame.y)


def test_review_weights_without_winner_changes_preserve_evaluation(source):
    result = evaluate_ensemble(source, tuple(representatives(source)[1]), 'mean')
    evidence = source.tuning_results
    evidence['review']['weights'] = [0, 1, 0]
    updated = source.with_tuning_results(evidence)
    assert updated.with_ensemble(result).ensemble_results is not None


def test_holdout_and_prediction_rows_never_enter_ensemble_fits():
    from evaluation_plan import level_key
    from model_review import completed_evidence
    from pipeline_search import fresh, prepare, run
    from roles import RoleMap
    source = example_source()
    frame = source.clean_frame.copy()
    frame['partition'] = ['train'] * 20 + ['valid'] * 10 + ['held'] * 5 + ['predict'] * 5
    frame.loc[35:, 'y'] = np.nan
    source = replace(source, _df=frame, _clean_df=frame,
        _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y'], 'partition': ['partition']}),
        _tuning_results=None)
    plan = build_plan(source, goal='observations', partition_map={level_key(k): v for k, v in
        [('train', 'development'), ('valid', 'validation'), ('held', 'holdout'), ('predict', 'prediction')]})
    assert plan.ready, plan.errors
    source = source.with_evaluation_plan(plan, card='obs_resample')
    problem = prepare(replace(source, _selected_candidate=None))
    state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    evidence = completed_evidence(problem, state)
    chosen = decisions(candidate_frame(evidence)[0], problem.specification['metric'], [1, 0, 0], 1e-12)
    evidence['review'] = {'representatives': {key: value - 1 for key, value in chosen['methods'].items()}}
    source = source.with_tuning_results(evidence)
    result = evaluate_ensemble(source, tuple(representatives(source)[1]), 'mean')
    assert result.splits == ((tuple(range(20)), tuple(range(20, 30))),)
    exported = source.with_ensemble(result)
    assert exported.evaluation_plan.holdout == tuple(range(30, 35))
    assert exported.evaluation_plan.prediction == tuple(range(35, 40))
    assert not exported.evaluation_plan.holdout_used
    for _, model in result.fitted[0].estimators_:
        assert model.named_steps['cand_methods'].named_steps['scale'].mean_[0] == pytest.approx(9.5)


def test_evaluation_identity_tracks_selected_inputs_only(source):
    from ensemble import evaluation_signature
    from model_review import CandidatePipelines
    selected = tuple(representatives(source)[1])[:2]
    signature = evaluation_signature(source, selected, 'mean')
    evidence = source.tuning_results
    evidence['state']['trials'].append(dict(evidence['state']['trials'][-1], id=999))
    evidence['state']['elapsed'] = 999
    updated = source.with_tuning_results(evidence)
    assert evaluation_signature(updated, selected, 'mean') == signature
    # A different overall winner need not be a selected ensemble member.
    updated = updated.with_selected_model(evidence['state']['trials'][-2]['id'])
    assert evaluation_signature(updated, selected, 'mean') == signature
    assert evaluation_signature(updated, selected, 'median') != signature
    with pytest.raises(ValueError, match='at least two'):
        evaluation_signature(updated, selected[:1], 'mean')
    pipelines = evidence['pipelines']
    method, _ = pipelines.choices[str(evidence['review']['representatives'][selected[0]])]
    templates = dict(pipelines.templates)
    changed = clone(templates[method])
    changed.set_params(cand_methods__estimator__alpha=1234)
    templates[method] = changed
    evidence['pipelines'] = CandidatePipelines(templates, pipelines.choices)
    updated = source.with_tuning_results(evidence)
    assert evaluation_signature(updated, selected, 'mean') != signature


def test_snapshot_context_ignores_winner_progress_but_tracks_data_and_resampling_seed(source):
    from ensemble import evaluation_context_signature
    context = evaluation_context_signature(source)
    evidence = source.tuning_results
    evidence['review']['representatives'].pop(next(iter(evidence['review']['representatives'])))
    assert evaluation_context_signature(source.with_tuning_results(evidence)) == context
    evidence['state']['specification']['seed'] += 1
    assert evaluation_context_signature(source.with_tuning_results(evidence)) != context
    frame = source.clean_frame.copy()
    frame.loc[0, 'x'] += 10
    changed = replace(source, _df=frame, _clean_df=frame).with_tuning_results(source.tuning_results)
    assert evaluation_context_signature(changed) != context
