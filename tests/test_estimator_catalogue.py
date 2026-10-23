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
