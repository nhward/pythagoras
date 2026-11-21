"""Held-out assessment selection, fit reuse, scoring and display contracts."""
import threading
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from candidate_assessment import (
    AssessmentWaiting, assess_candidate, assessment_figure, assessment_signature,
    assessment_table, group_masks, group_values, grouping_columns, roc_splits,
)
from ensemble import evaluate_ensemble, representatives
from ensemble_scenario import example_source
from evaluation_plan import build_plan
from fit_service import FitService, MemoryFitService
from model_review import completed_evidence
from pipeline_search import fresh, prepare, run
from roles import RoleMap
from sklearn.base import BaseEstimator, ClassifierMixin


@pytest.fixture(scope='module')
def regression():
    return example_source(count=2)


@pytest.fixture(scope='module')
def classification():
    return example_source('classification', count=2)


def test_requires_weighted_winner_and_ignores_unrelated_training_progress(regression):
    with pytest.raises(AssessmentWaiting, match='weighted'):
        assess_candidate(replace(regression, _tuning_results=None))
    evidence = regression.tuning_results
    evidence['review']['winner_trial'] = None
    with pytest.raises(AssessmentWaiting, match='weighted'):
        assess_candidate(regression.with_tuning_results(evidence))
    evidence = regression.tuning_results
    signature = assessment_signature(regression)
    evidence['state']['trials'][1]['note'] = 'An unrelated progress update'
    assert assessment_signature(regression.with_tuning_results(evidence)) == signature


def test_regression_fit_reuse_alignment_reference_limits_and_statistics(regression, tmp_path):
    source = regression
    with_service = FitService(cpus=2, directory=tmp_path)
    try:
        first = assess_candidate(source, fit_service=with_service)
        warm = assess_candidate(source, fit_service=with_service)
        assert first.new_fits == first.splits == 5
        assert warm.new_fits == 0
        assert warm.reused_fits == 5
        assert source is regression
        np.testing.assert_array_equal(first.events.actual, source.clean_frame.y.iloc[first.events.row])
        table = assessment_table(first)
        error = first.events.predicted - first.events.actual
        assert table.loc[0, 'Mean error (bias)'] == pytest.approx(error.mean())
        assert table.loc[0, 'Error variance'] == pytest.approx(error.var(ddof=0))
        assert table.loc[0, 'RMSE'] ** 2 == pytest.approx(table.loc[0, 'Error variance'] + table.loc[0, 'Mean error (bias)'] ** 2)
        figure = assessment_figure(first)
        assert figure.layout.yaxis.scaleanchor == 'x'
        assert figure.layout.xaxis.range == figure.layout.yaxis.range
        for name, quantile in [('Error P2.5', .025), ('Error P97.5', .975)]:
            trace = next(t for t in figure.data if t.name == name)
            np.testing.assert_allclose(np.asarray(trace.y) - trace.x, np.quantile(error, quantile))
        with with_service.runtime.database() as db:
            assert not db.execute("SELECT 1 FROM pins WHERE owner LIKE 'assessment:%'").fetchone()
    finally:
        with_service.close()


def test_single_model_reuses_upstream_training_artifacts(regression, tmp_path):
    # Training and assessment must hash the same prepared complete pipelines.
    from pipeline_search import evaluate_fold
    service = FitService(cpus=2, directory=tmp_path)
    try:
        service.enable()
        problem = prepare(regression)
        problem.fit_runtime = service.runtime
        for fold in range(len(problem.splits)):
            evaluate_fold(problem, {}, fold)
        result = assess_candidate(regression, fit_service=service)
        assert result.new_fits == 0
        assert result.reused_fits == len(problem.splits)
    finally:
        service.close()


def test_committed_ensemble_precedence_and_stale_ensemble_waits(regression):
    evaluation = evaluate_ensemble(regression, tuple(representatives(regression)[1]), 'mean')
    source = regression.with_ensemble(evaluation)
    result = assess_candidate(source)
    assert result.label.startswith('Ensemble')
    assert result.new_fits == 0
    assert result.reused_fits == 10
    for fold, (_, validation) in enumerate(evaluation.splits):
        actual = result.events.loc[result.events.split == fold, 'predicted']
        np.testing.assert_allclose(actual, evaluation.fitted[fold].predict(source.clean_frame.iloc[list(validation)]))
    frame = source.clean_frame.assign(x=source.clean_frame.x + 1)
    stale = replace(source, _df=frame, _clean_df=frame)
    with pytest.raises(AssessmentWaiting, match='ensemble'):
        assess_candidate(stale)


def test_hard_vote_roc_uses_member_vote_shares(classification):
    evaluation = evaluate_ensemble(classification, tuple(representatives(classification)[1]), 'hard')
    result = assess_candidate(classification.with_ensemble(evaluation))
    assert result.score_kind == 'Vote share'
    assert result.new_fits == 0
    assert set(np.unique(result.scores)) <= {0., .5, 1.}
    np.testing.assert_allclose(result.scores.sum(axis=1), 1.)
    for fold, (_, validation) in enumerate(evaluation.splits):
        model = evaluation.fitted[fold]
        votes = np.asarray([m.predict(classification.clean_frame.iloc[list(validation)]) for _, m in model.estimators_])
        np.testing.assert_allclose(result.scores[result.events.split == fold, 1], (votes == result.classes[1]).mean(axis=0))
    assert len(roc_splits(result, np.ones(len(result.events), dtype=bool), 1)[0]) == 5
    table = assessment_table(result, class_index=1)
    assert table.loc[0, 'ROC splits'] == 5
    assert 'Mean probability error' not in table
    assert any(t.name and 'AUC' in t.name for t in assessment_figure(result, class_index=1).data)
    heatmap = assessment_figure(result, chart='confusion').data[0]
    assert heatmap.type == 'heatmap'
    np.testing.assert_allclose(np.asarray(heatmap.z).sum(axis=1), 100.)


def test_probability_residuals_and_missing_class_groups(classification):
    result = assess_candidate(classification)
    assert result.score_kind == 'Probability'
    result.groups = pd.DataFrame({'group': classification.clean_frame.y.map({0: 'negative', 1: 'positive'})})
    table = assessment_table(result, 'group', class_index=1)
    probability_error = result.scores[:, 1] - (result.events.actual.to_numpy() == result.classes[1])
    assert table.loc[0, 'Mean probability error'] == pytest.approx(probability_error.mean())
    assert table.loc[0, 'Probability error variance'] == pytest.approx(probability_error.var())
    assert table.loc[1:, 'ROC splits'].eq(0).all()
    assert table.loc[1:, 'AUC'].isna().all()


class LabelsOnly(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.majority_ = self.classes_[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.repeat(self.majority_, len(X))


def test_labels_only_classification_falls_back_to_confusion(classification):
    results = classification.tuning_results
    winner = results['review']['winner_trial']
    method = results['pipelines'].choices[str(winner)][0]
    results['pipelines'].templates[method].set_params(cand_methods__estimator=LabelsOnly())
    source = classification.with_tuning_results(results)
    result = assess_candidate(source)
    assert result.scores is None
    assert assessment_figure(result).data[0].type == 'heatmap'
    assert assessment_table(result).AUC.isna().all()


def test_group_roles_numeric_bins_missing_and_duplicate_indices(regression):
    frame = regression.clean_frame.assign(stratum=np.arange(40, dtype=float), sensitive=['a', 'b'] * 20, ignored=np.arange(40))
    frame.loc[0, 'stratum'] = np.nan
    frame.index = [0] * 40
    roles = RoleMap.from_primitive({'predictor': ['x'], 'target': ['y'], 'stratifier': ['stratum'], 'sensitive': ['sensitive'], 'identifier': ['ignored']})
    source = replace(regression, _df=frame, _clean_df=frame, _roles=roles)
    # Reattach compatible evidence to the deliberately changed upstream context.
    source = source.with_evaluation_plan(build_plan(source, strategy='kfold', objective='independent', folds=5, stratifier=None), card='obs_resample')
    source = source.with_tuning_results(regression.tuning_results)
    assert grouping_columns(source) == ['stratum', 'sensitive']
    result = assess_candidate(source)
    labels = group_values(result, 'stratum')
    assert labels.nunique() == 6
    assert labels[result.events.row == 0].eq('<Missing>').all()
    assert group_values(result, 'sensitive').tolist() == frame.sensitive.iloc[result.events.row].tolist()
    table = assessment_table(result, 'sensitive')
    assert table.loc[0, 'Predictions'] == table.loc[1:, 'Predictions'].sum()
    assert len(list(group_masks(result, 'sensitive', 'sensitive'))) == 3


def test_repeated_splits_keep_validation_events(regression):
    source = regression.with_evaluation_plan(build_plan(regression, strategy='kfold', objective='independent', folds=3, repeats=2), card='obs_resample')
    problem = prepare(source, method='grid')
    state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    evidence = completed_evidence(problem, state)
    evidence['review'] = {'winner_trial': state['trials'][0]['id']}
    result = assess_candidate(source.with_tuning_results(evidence))
    assert len(result.events) == 80
    assert result.events.row.nunique() == 40
    assert result.events.groupby('row').size().eq(2).all()
    assert assessment_table(result).loc[0, 'Observations'] == 40


def test_cancellation_does_not_fit(regression):
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InterruptedError):
        assess_candidate(regression, cancelled=cancelled, fit_service=MemoryFitService())


def test_multiclass_roc_aligns_classes_and_selects_one_versus_rest(classification):
    frame = classification.clean_frame.assign(y=np.arange(40) % 3)
    source = replace(classification, _df=frame, _clean_df=frame).with_tuning_results(classification.tuning_results)
    result = assess_candidate(source)
    assert result.scores.shape == (40, 3)
    np.testing.assert_allclose(result.scores.sum(axis=1), 1.)
    for index in range(3):
        assert len(roc_splits(result, np.ones(40, dtype=bool), index)[0]) == 5
        assert assessment_table(result, class_index=index).loc[0, 'ROC splits'] == 5
    assert np.asarray(assessment_figure(result, chart='confusion').data[0].z).shape == (3, 3)


def test_decision_scores_are_rankings_not_probability_residuals(classification):
    from sklearn.svm import LinearSVC

    results = classification.tuning_results
    method = results['pipelines'].choices[str(results['review']['winner_trial'])][0]
    results['pipelines'].templates[method].set_params(cand_methods__estimator=LinearSVC(random_state=42))
    result = assess_candidate(classification.with_tuning_results(results))
    assert result.score_kind == 'Decision score'
    assert assessment_table(result, class_index=1).loc[0, 'ROC splits'] == 5
    assert 'Mean probability error' not in assessment_table(result)


def test_generated_stratifiers_use_row_aligned_preview(regression):
    preview = regression.frame.assign(cluster__example=['one', 'two'] * 20)
    roles = regression.role_map.to_primitive()
    roles['stratifier'] = ['cluster__example']
    source = replace(regression, _df=preview, _roles=RoleMap.from_primitive(roles))
    source = source.with_evaluation_plan(build_plan(source, strategy='kfold', objective='independent', folds=5, stratifier=None), card='obs_resample')
    source = source.with_tuning_results(regression.tuning_results)
    assert grouping_columns(source) == ['cluster__example']
    result = assess_candidate(source)
    assert group_values(result, 'cluster__example').tolist() == preview.cluster__example.iloc[result.events.row].tolist()
    assert len(assessment_table(result, 'cluster__example')) == 3
