"""Static catalogue integrity and conservative capability filtering."""
import json
import subprocess
import sys
from copy import deepcopy

import jsonschema
import pytest
from estimator_catalogue import DEFAULT_PATH, find_profiles, load_catalogue


@pytest.fixture
def catalogue():
    return load_catalogue()


@pytest.mark.unit
def test_benchmark_purpose_matches_source(catalogue):
    source = json.loads(DEFAULT_PATH.with_name('benchmark_methods.json').read_text())
    actual = {e['id']: e['purpose_reason'] for e in catalogue['estimators']
              if e.get('purpose') == 'benchmark_only'}
    assert actual == source


@pytest.mark.unit
def test_schema_and_profile_references(catalogue):
    schema = json.loads(DEFAULT_PATH.with_name('schema.json').read_text())
    jsonschema.validate(catalogue, schema)
    entries = {e['id']: e for e in catalogue['estimators']}
    profiles = {p['id']: p for p in catalogue['profiles']}
    assert len(entries) == len(catalogue['estimators'])
    assert len(profiles) == len(catalogue['profiles'])
    for entry in entries.values():
        for identifier in entry['achievable_profile_ids']:
            assert profiles[identifier]['estimator'] == entry['id']
    for profile in profiles.values():
        if profile['validation'] == 'passed':
            assert profile['version_match']
            assert profile['id'] in entries[profile['estimator']]['achievable_profile_ids']
            assert profile['validation_details']['operations']


@pytest.mark.unit
def test_supersession_source_references_and_family_limits(catalogue):
    source = json.loads(DEFAULT_PATH.with_name('supersession.json').read_text())
    actual = {e['id']: e['superseded_by'] for e in catalogue['estimators'] if 'superseded_by' in e}
    assert actual == source['relationships']
    entries = {e['id'] for e in catalogue['estimators']}
    assert all(r['method'] in entries for r in actual.values())
    ridge = actual['sklearn.linear_model.Ridge']
    assert ridge['relationship'] == 'objective_family'
    assert 'dedicated Ridge solver' in ridge['caveats']
    assert 'sklearn.linear_model.BayesianRidge' not in actual
    assert 'sklearn.svm.LinearSVC' not in actual
    assert 'sklearn.ensemble.RandomForestRegressor' not in actual


@pytest.mark.unit
@pytest.mark.parametrize('batch', ['priority_2026_10_24', 'priority_2_2026_10_24',
                                  'priority_3_2026_10_24', 'priority_4_2026_10_24',
                                  'priority_5_2026_10_24'])
def test_priority_batch_has_scoped_validation_evidence(catalogue, batch):
    profiles = [p for p in catalogue['profiles'] if p.get('probe', {}).get('batch') == batch]
    assert len(profiles) == 10
    for profile in profiles:
        assert profile['validation'] == 'passed' and profile['version_match']
        operations = profile['validation_details']['operations']
        assert {'clone_get_set_params', 'pipeline_three_fold_held_out_prediction',
                'fit_predict_finite', 'no_convergence_warnings'} <= set(operations)
        if 'class_probability' in profile['outputs']:
            assert 'held_out_class_probabilities_normalized' in operations
        if profile['capabilities']['sample_weight_fit']:
            assert 'fit_accepts_nonuniform_sample_weights' in operations
        if profile['capabilities']['categorical_predictors']:
            assert 'predict_unseen_category' in operations
    if batch == 'priority_5_2026_10_24':
        assert all(p['construction']['preparation'] == 'numeric' for p in profiles)
        assert sum(p['capabilities']['sample_weight_fit'] for p in profiles) == 7
        assert sum(p['probe']['probabilities'] for p in profiles) == 4
        for profile in profiles:
            if profile['estimator'].endswith(('PoissonRegressor', 'GammaRegressor')):
                assert 'held_out_predictions_positive' in profile['validation_details']['operations']
            if profile['estimator'].endswith(('MultinomialNB', 'ComplementNB')):
                assert profile['probe']['fixture'] == 'nonnegative_counts'
        return
    if batch == 'priority_4_2026_10_24':
        assert sum(p['capabilities']['sample_weight_fit'] for p in profiles) == 5
        assert sum(p['probe']['probabilities'] for p in profiles) == 3
        for profile in profiles:
            if profile['estimator'].endswith(('RidgeCV', 'LogisticRegressionCV', 'RidgeClassifierCV', 'LassoLarsIC')):
                assert profile['construction']['preparation'] == 'numeric'
                assert not profile['capabilities']['missing_numeric_predictors']
            if profile['estimator'].endswith(('RidgeClassifier', 'RidgeClassifierCV')):
                assert profile['outputs'] == ['class_prediction']
        return
    if batch == 'priority_3_2026_10_24':
        assert all(p['construction']['preparation'] == 'impute_onehot' for p in profiles)
        assert sum(p['capabilities']['sample_weight_fit'] for p in profiles) == 7
        assert sum(p['probe']['probabilities'] for p in profiles) == 5
        return
    numeric_method = 'ElasticNetCV' if batch == 'priority_2026_10_24' else 'QuadraticDiscriminantAnalysis'
    numeric = next(p for p in profiles if p['estimator'].endswith(numeric_method))
    assert numeric['construction']['preparation'] == 'numeric'
    assert not numeric['capabilities']['missing_numeric_predictors']


@pytest.mark.unit
def test_transformer_batch_has_scoped_validation_evidence(catalogue):
    profiles = [p for p in catalogue['profiles']
                if p.get('probe', {}).get('batch') == 'priority_6_2026_10_25']
    assert len(profiles) == 20
    assert len({p['estimator'] for p in profiles}) == 20
    for profile in profiles:
        assert profile['validation'] == 'passed' and profile['version_match']
        assert profile['task'] == 'transformer' and profile['outputs'] == []
        assert profile['construction']['preparation'] == 'numeric'
        assert not profile['capabilities']['incremental_training']
        assert not profile['capabilities']['missing_numeric_predictors']
        operations = set(profile['validation_details']['operations'])
        assert {'clone_get_set_params', 'pipeline_three_fold_held_out_prediction',
                'fit_transform_row_count', 'fit_transform_finite_features',
                'held_out_transform_finite_consistent_features',
                'no_convergence_warnings'} <= operations
        if profile['capabilities']['sample_weight_fit']:
            assert 'fit_accepts_nonuniform_sample_weights' in operations
    weighted = {p['estimator'] for p in profiles if p['capabilities']['sample_weight_fit']}
    assert weighted == {'sklearn.preprocessing.StandardScaler', 'sklearn.preprocessing.SplineTransformer'}


@pytest.mark.unit
@pytest.mark.parametrize('failure', ['nonfinite', 'changed_columns', 'dropped_row'])
def test_transform_probe_rejects_invalid_features(catalogue, monkeypatch, failure):
    import runpy
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin

    class BrokenTransformer(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            rows = len(X) - (failure == 'dropped_row')
            cols = 3 if failure == 'changed_columns' and len(X) == 30 else 2
            return np.full((rows, cols), np.nan if failure == 'nonfinite' else 1.0)

    script = DEFAULT_PATH.parents[3] / 'scripts/probe_estimator_catalogue.py'
    probe = runpy.run_path(str(script))['probe']
    profile = deepcopy(next(p for p in catalogue['profiles']
                            if p['estimator'] == 'sklearn.decomposition.PCA'))
    profile['probe']['pipeline_cross_validation'] = False
    monkeypatch.setitem(probe.__globals__, 'construct',
                        lambda profile, X: (BrokenTransformer(), X, {}, {}))
    with pytest.raises(AssertionError):
        probe(profile)


@pytest.mark.unit
def test_probe_batch_preserves_unselected_results(catalogue, tmp_path):
    from pathlib import Path
    selected = next(p for p in catalogue['profiles'] if p['estimator'] == 'sklearn.linear_model.HuberRegressor')
    selected['probe']['batch'] = 'test_single_recipe'
    before = {p['id']: deepcopy(p) for p in catalogue['profiles'] if p['id'] != selected['id']}
    path = tmp_path / 'catalogue.json'
    path.write_text(json.dumps(catalogue))
    script = Path(__file__).resolve().parents[1] / 'scripts/probe_estimator_catalogue.py'
    result = subprocess.run([sys.executable, str(script), '--catalogue', str(path),
                             '--batch', 'test_single_recipe'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    after = json.loads(path.read_text())
    assert {p['id']: p for p in after['profiles'] if p['id'] != selected['id']} == before


@pytest.mark.unit
@pytest.mark.parametrize('target', ['absent.Estimator', 'sklearn.linear_model.Ridge'])
def test_invalid_supersession_rejected(catalogue, tmp_path, target):
    ridge = next(e for e in catalogue['estimators'] if e['id'] == 'sklearn.linear_model.Ridge')
    ridge['superseded_by']['method'] = target
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(catalogue))
    with pytest.raises(ValueError, match='[Ss]upersession|supersession'):
        load_catalogue(path)


@pytest.mark.unit
def test_mixed_missing_regression_search(catalogue):
    matches = find_profiles(catalogue, task='regressor', required=(
        'categorical_predictors', 'missing_numeric_predictors',
        'missing_categorical_predictors', 'sample_weight_fit'))
    identifiers = {p['id'] for p in matches}
    assert 'sklearn.ensemble.HistGradientBoostingRegressor:native_mixed_missing' in identifiers
    assert 'sklearn.linear_model.Ridge:mixed_missing_via_preprocessing' in identifiers
    assert not any(p['estimator'] == 'sklearn.svm.SVR' for p in matches)
    assert all(p['task'] == 'regressor' for p in matches)


@pytest.mark.unit
def test_unknown_failed_and_version_mismatches_are_not_candidates(catalogue):
    specimen = deepcopy(next(p for p in catalogue['profiles'] if p['validation'] == 'passed'))
    catalogue['profiles'] = [specimen]
    specimen['capabilities']['new_unknown'] = None
    assert not find_profiles(catalogue, required=['new_unknown'])
    assert not find_profiles(catalogue, required=['not_a_capability'])
    specimen['validation'] = 'failed'
    assert not find_profiles(catalogue)
    specimen['validation'] = 'passed'
    specimen['version_match'] = False
    assert not find_profiles(catalogue, validated_only=False)


@pytest.mark.unit
def test_deployment_versions_include_core_dependencies(catalogue):
    profile = next(p for p in catalogue['profiles'] if p['validation'] == 'passed')
    catalogue['profiles'] = [profile]
    entry = next(e for e in catalogue['estimators'] if e['id'] == profile['estimator'])
    versions = profile['inspected_dependencies'] | {entry['package']: entry['package_version']}
    assert find_profiles(catalogue, package_versions=versions) == [profile]
    versions['numpy'] = '0.0.0'
    assert not find_profiles(catalogue, package_versions=versions)
    assert not find_profiles(catalogue, package_versions={})


@pytest.mark.unit
def test_query_does_not_import_estimator_libraries():
    # A fresh interpreter proves JSON searching does not trigger sklearn imports.
    source = """
import sys
from estimator_catalogue import load_catalogue, find_profiles
assert find_profiles(load_catalogue(), task='regressor')
assert not {'sklearn', 'xgboost', 'catboost', 'lightgbm'} & set(sys.modules)
"""
    import os
    environment = dict(os.environ, PYTHONPATH=str(DEFAULT_PATH.parents[2]))
    subprocess.run([sys.executable, '-c', source], env=environment, check=True)


@pytest.mark.unit
def test_judgements_are_separate_from_benchmarks(catalogue):
    for profile in catalogue['profiles']:
        assert profile['subjective_assessment']['kind'] == 'expert_judgement_not_benchmark'
        assert profile['scalability']['benchmark'] is None
        assert profile['weighting']['better_than_most'] is None
        assert profile['sources']


@pytest.mark.unit
def test_generated_profiles_match_reviewed_source(catalogue):
    source = json.loads(DEFAULT_PATH.with_name('profiles.json').read_text())
    generated = {profile['id']: profile for profile in catalogue['profiles']}
    for profile in source:
        assert all(generated[profile['id']][key] == value for key, value in profile.items())


@pytest.mark.unit
def test_specialist_source_and_evidence(catalogue):
    from estimator_catalogue import assess_profiles, coverage
    assert catalogue['specialists'] == json.loads(DEFAULT_PATH.with_name('specialists.json').read_text())
    rows = assess_profiles(catalogue, task='survival', required=['right_censoring'],
                           available_requirements=['event_time_target'], package_versions={})
    assert rows and all(not r['selectable'] for r in rows)
    assert any(r['compatibility'] == 'compatible' for r in rows)
    assert all(r['availability'] == 'not_installed' for r in rows)
    assert coverage(catalogue)['survival']['probed_recipes'] == 0


@pytest.mark.unit
def test_documented_rule_models_remain_conservative(catalogue):
    from estimator_catalogue import assess_profiles

    recipes = {p['estimator']: p for p in catalogue['specialists']}
    expected = {'cubist.Cubist', 'skrules.SkopeRules'} | {
        f'lineartree.Linear{family}{kind}'
        for family in ('Tree', 'Forest', 'Boost')
        for kind in ('Regressor', 'Classifier')
    }
    assert expected <= recipes.keys()
    assert not recipes['lineartree.LinearForestClassifier']['capabilities']['multiclass']
    rows = assess_profiles(catalogue, task='classification', required=['sample_weight_fit'])
    skope = next(row for row in rows if row['estimator'] == 'skrules.SkopeRules')
    assert skope['compatibility'] == 'incompatible'
    for task in ('regression', 'classification'):
        rows = assess_profiles(catalogue, task=task)
        additions = [row for row in rows if row['id'] in {recipes[name]['id'] for name in expected}]
        assert additions
        assert all(row['evidence'] == 'documentation_only' and not row['selectable']
                   for row in additions)


@pytest.mark.unit
def test_structural_requirements_are_not_inferred_from_numeric_support(catalogue):
    from estimator_catalogue import assess_profiles
    rows = assess_profiles(catalogue, task='regression', structure=['spatial'])
    forest = next(r for r in rows if r['estimator'] == 'sklearn.ensemble.RandomForestRegressor')
    assert forest['compatibility'] == 'unknown'
    spatial = next(r for r in rows if r['estimator'] == 'spreg.ML_Lag')
    assert spatial['compatibility'] == 'incompatible'
    assert 'Requires spatial_weights' in spatial['reasons']
    rows = assess_profiles(catalogue, task='regression', structure=['spatial'],
                           available_requirements=['spatial_weights'])
    assert next(r for r in rows if r['estimator'] == 'spreg.ML_Lag')['compatibility'] == 'compatible'


@pytest.mark.unit
def test_cached_snapshots_are_isolated_and_refresh(tmp_path, catalogue):
    path = tmp_path / 'catalogue.json'
    path.write_text(json.dumps(catalogue))
    first = load_catalogue(path)
    first['profiles'].clear()
    assert load_catalogue(path)['profiles']
    catalogue['catalogue_version'] = '1.1.123'
    path.write_text(json.dumps(catalogue))
    assert load_catalogue(path)['catalogue_version'] == '1.1.123'
    catalogue['schema_version'] = '99.0.0'
    path.write_text(json.dumps(catalogue))
    with pytest.raises(ValueError, match='Unsupported'):
        load_catalogue(path)


@pytest.mark.unit
def test_proxy_summary_excludes_outcomes():
    import pandas as pd
    from estimator_catalogue import requirements_from_proxy
    from modelling_context import ModellingContext
    from proxy_data import proxy_data
    frame = pd.DataFrame({'duration': [1, 2], 'event': [1, 0],
                          'x': [1., None], 'category': pd.Categorical(['a', None])})
    source = proxy_data(frame).with_modelling_context(ModellingContext(task='survival', duration='duration', event='event'))
    summary = requirements_from_proxy(source, use_weights=True)
    assert summary['predictors'] == 2
    assert summary['task'] == 'survival'
    assert set(summary['required']) == {'numeric_predictors', 'categorical_predictors',
        'missing_numeric_predictors', 'missing_categorical_predictors', 'sample_weight_fit', 'right_censoring'}


@pytest.mark.unit
def test_assessment_deployment_and_task_aliases(catalogue):
    from estimator_catalogue import assess_profiles
    p = next(p for p in catalogue['profiles'] if p['task'] == 'regressor' and p['validation'] == 'passed')
    entry = next(e for e in catalogue['estimators'] if e['id'] == p['estimator'])
    versions = p['inspected_dependencies'] | {entry['package']: entry['package_version']}
    rows = assess_profiles(catalogue, task='regression', package_versions=versions)
    assert next(r for r in rows if r['id'] == p['id'])['selectable']
    versions['numpy'] = 'wrong'
    rows = assess_profiles(catalogue, task='regression', package_versions=versions)
    assert not next(r for r in rows if r['id'] == p['id'])['selectable']


def test_input_requirements_do_not_prescribe_tested_preparation():
    from estimator_catalogue import input_assessment
    profile = {'construction': {'preparation': 'impute_onehot'}}
    estimator = {'default_tags': {'input_tags': {'allow_nan': False}}}
    assessment = input_assessment(profile, estimator,
        ['categorical_predictors', 'missing_numeric_predictors'])
    assert assessment['readiness'] == 'Requires preparation'
    assert 'No missing predictor values' in assessment['requirements']
    assert 'imput' not in assessment['requirements'].lower()
    assert 'onehot' not in assessment['requirements'].lower()
    assert input_assessment(profile, estimator, ['numeric_predictors'])['readiness'] == 'Meets reviewed input requirements'
    assert input_assessment(profile, {}, [])['readiness'] == 'Unreviewed'


def test_configured_categorical_support_overrides_default_tags():
    from estimator_catalogue import input_assessment
    profile = {'construction': {'preparation': 'pandas_categories'},
               'capabilities': {'missing_numeric_predictors': True, 'missing_categorical_predictors': True}}
    assessment = input_assessment(profile, {'default_tags': {'input_tags': {'categorical': False}}},
        ['numeric_predictors', 'missing_numeric_predictors'])
    assert assessment['readiness'] == 'Meets reviewed input requirements'
    assert 'No missing' not in assessment['requirements']
    assert input_assessment({}, {}, [])['readiness'] == 'Unreviewed'
