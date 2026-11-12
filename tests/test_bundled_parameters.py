"""Conditional spaces follow actual nested configuration owned by each bundle."""
import json

import numpy as np
import pandas as pd
import pytest
from cards.miss_impute import ImputationStep
from cards.var_transform import VariableTransformStep
from ClusterMembershipsTransformer import ClusterMembershipsTransformer
from ClusterMembershipTransformer import ClusterMembershipTransformer
from FeatureSelectionTransformer import FeatureSelectionTransformer
from OrderedEncodingTransformer import OrderedEncodingTransformer
from pipeline_search import (
    accepted_parameters,
    fresh,
    grid_coverage,
    prepare,
    run,
    validate_checkpoint,
)
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from TargetEncodingTransformer import TargetEncodingTransformer
from test_pipeline_search import source
from TextEncodingTransformer import TextEncodingTransformer
from tuning import compile_spec, discover, discover_source, search_values

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('estimator,path,value,legacy', [
    (ImputationStep(('x',), ('x',)), 'knn__n_neighbors', 3, 'neighbors'),
    (ImputationStep(('x',), ('x',)), 'iterative__max_iter', 4, 'iterations'),
    (OrderedEncodingTransformer(('x',)), 'polynomial__degree', 2, 'degree'),
    (TargetEncodingTransformer(('x',), target='y'), 'smooth__smoothing', 20., 'smoothing'),
    (TextEncodingTransformer(('x',)), 'lsa__components', 3, 'components'),
    (TextEncodingTransformer(('x',)), 'vectorizer__ngram_max', 2, 'ngram_max'),
    (ClusterMembershipTransformer(('x',)), 'partition__center', 'medoids', 'centre'),
    (ClusterMembershipsTransformer(('x',)), 'density__min_points', 8, 'density_min_points'),
    (FeatureSelectionTransformer(('x',)), 'boruta__iterations', 12, 'iterations'),
    (FeatureSelectionTransformer(('x',)), 'groot__folds', 3, 'folds'),
    (VariableTransformStep(('x',)), 'location_scale__robust', True, 'robust'),
])
def test_nested_configuration_clones_and_legacy_aliases_still_work(estimator, path, value, legacy):
    changed = clone(estimator).set_params(**{path: value})
    assert changed.parameter_value(legacy) == value
    assert clone(changed).get_params()[path] == value
    changed = clone(estimator).set_params(**{legacy: value})
    assert changed.get_params()[path] == value
    rows = {r['path']: r for r in discover(Pipeline([('step', changed)]))}
    assert f'step__{path}' in rows
    assert f'step__{legacy}' not in rows
    assert f'step__{legacy}' in rows[f'step__{path}']['legacy_paths']


def drafts_for(rows, choices):
    return {r['path']: dict(r['definition'], values=choices[r['path']], mode='values', active=True, owner=r['owner'])
            for r in rows if r['path'] in choices}


def imputation_source(method='simple'):
    data = source()
    return data.with_pipeline_step(ImputationStep(('x',), ('x',), method=method),
                                   name='miss_impute', preview_frame=data.frame)


def configured(method='simple', tune_method=True):
    data = imputation_source(method)
    rows = discover_source(data)
    choices = {'miss_impute__knn__n_neighbors': [3, 4, 5, 6]}
    if tune_method:
        choices['miss_impute__method'] = ['knn', 'simple', 'iterative']
    spec = compile_spec(rows, drafts_for(rows, choices))
    return data.with_tuning_spec(spec)


def test_six_combinations_and_fixed_method_counts():
    assert configured().tuning_spec['combinations'] == 6
    assert configured('knn', False).tuning_spec['combinations'] == 4
    assert configured('simple', False).tuning_spec['combinations'] == 1
    assert configured('iterative', False).tuning_spec['combinations'] == 1
    row = next(r for r in discover_source(configured()) if r['path'].endswith('__n_neighbors'))
    assert row['name'] == 'Neighbors'
    assert search_values(row['definition']) == [3, 4, 5, 6]


@pytest.mark.parametrize('method', ['grid', 'random', 'tpe', 'skopt'])
def test_all_searches_evaluate_six_unique_active_configurations_and_resume(method):
    if method == 'tpe':
        pytest.importorskip('optuna')
    if method == 'skopt':
        pytest.importorskip('skopt')
    problem = prepare(configured(), method=method)
    assert problem.specification['combinations'] == 6
    initial = run(problem, fresh(problem), trials=2, publish=lambda s: None, pause=lambda: False)
    result = run(problem, json.loads(json.dumps(initial)), trials=0, publish=lambda s: None, pause=lambda: False)
    assert result['status'] == 'exhausted'
    assert len(result['trials']) == 6  # The simple baseline is already in the grid.
    assert grid_coverage(result)['complete'] == 6
    validate_checkpoint(problem, result)
    for trial in result['trials']:
        assert trial['state'] == 'complete'
        params = trial['params']
        assert ('miss_impute__knn__n_neighbors' in params) == (params['miss_impute__method'] == 'knn')
    chosen = next(t for t in result['trials'] if t['params'].get('miss_impute__knn__n_neighbors') == 3)
    output = configured().with_tuned_parameters(accepted_parameters(problem, result, chosen['id']),
        fingerprint=problem.specification['fingerprint'], trial=chosen['id'],
        metric=problem.specification['metric'], score=0.)
    fitted = output.pipeline_for_training().fit(output.clean_frame, output.frame.y)
    assert fitted.named_steps['miss_impute'].continuous_imputer_.n_neighbors == 3


def test_imputation_children_are_fitted_clones_and_inference_does_not_refit():
    frame = pd.DataFrame({'x': [0., 1., np.nan, 3., 4., 5.], 'z': [0., 1., 2., 3., 4., 5.]})
    step = ImputationStep(('x',), ('x', 'z'), method='knn').set_params(knn__n_neighbors=3)
    fitted = clone(step).fit(frame)
    assert not hasattr(step.knn, '_fit_X')
    assert fitted.continuous_imputer_.n_neighbors == 3
    assert fitted.transform(frame).x.notna().all()


def test_independent_imputation_instances_and_model_alternatives():
    from test_multi_model_search import candidates
    data = candidates()
    for name in ('first', 'second'):
        data = data.with_pipeline_step(ImputationStep(('x',), ('x',)), name=name, preview_frame=data.frame)
    rows = discover_source(data)
    choices = {f'{name}__method': ['simple', 'knn'] for name in ('first', 'second')}
    choices.update({f'{name}__knn__n_neighbors': [3, 4] for name in ('first', 'second')})
    spec = compile_spec(rows, drafts_for(rows, choices))
    assert spec['combinations'] == 18  # (1 + 2)² * two model methods
    problem = prepare(data.with_tuning_spec(spec), method='grid')
    assert problem.specification['combinations'] == 18


def test_shared_text_settings_and_transform_switch_conditions():
    for methods, expected in [(('sentiment',), 1), (('bow',), 2), (('bow', 'lsa'), 6)]:
        rows = discover(Pipeline([('text', TextEncodingTransformer(('x',), methods=methods))]))
        choices = {'text__vectorizer__ngram_max': [1, 2], 'text__lsa__components': [2, 3, 4]}
        assert compile_spec(rows, drafts_for(rows, choices))['combinations'] == expected
    rows = discover(Pipeline([('transform', VariableTransformStep(('x',)))]))
    choices = {'transform__scale': [False, True], 'transform__center': [False, True],
               'transform__location_scale__robust': [False, True]}
    assert compile_spec(rows, drafts_for(rows, choices))['combinations'] == 7


def test_saved_flat_names_migrate_and_old_spelling_is_accepted():
    from tuning import migrate_drafts
    step = ImputationStep(('x',), ('x',), neighbours=3)
    assert clone(step).knn.n_neighbors == 3
    assert step.set_params(neighbours=4).knn.n_neighbors == 4
    rows = discover(Pipeline([('miss_impute', step)]))
    old = {'miss_impute__neighbours': {'active': True, 'values': [3, 4]},
           'miss_impute__iterations': {'active': False, 'values': [5, 10]}}
    migrated = migrate_drafts(rows, old)
    assert migrated['miss_impute__knn__n_neighbors'] == old['miss_impute__neighbours']
    assert migrated['miss_impute__iterative__max_iter'] == old['miss_impute__iterations']
    assert old.keys() == {'miss_impute__neighbours', 'miss_impute__iterations'}
    canonical = {'active': True, 'values': [5, 6]}
    assert migrate_drafts(rows, {**old, 'miss_impute__knn__n_neighbors': canonical})['miss_impute__knn__n_neighbors'] == canonical


@pytest.mark.parametrize('execution', ['trials', 'folds'])
def test_conditional_search_works_with_parallel_execution(execution):
    problem = prepare(configured(), method='grid', execution=execution, workers=2)
    result = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    assert grid_coverage(result)['complete'] == 6
    validate_checkpoint(problem, json.loads(json.dumps(result)))


def test_sequence_selector_survives_json_checkpoints():
    data = source()
    data = data.with_pipeline_step(TextEncodingTransformer((), methods=('sentiment',)),
                                   name='text', preview_frame=data.frame)
    rows = discover_source(data)
    choices = {'text__lsa__components': [2, 3]}
    spec = compile_spec(rows, drafts_for(rows, choices))
    problem = prepare(data.with_tuning_spec(spec), method='grid')
    state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    assert grid_coverage(state)['complete'] == 1
    validate_checkpoint(problem, json.loads(json.dumps(state)))


@pytest.mark.parametrize('estimator,choices,expected', [
    (OrderedEncodingTransformer(('x',)),
     {'step__method': ['ordinal', 'polynomial'], 'step__polynomial__degree': [1, 2, 3]}, 4),
    (TargetEncodingTransformer(('x',), target='y'),
     {'step__method': ['auto', 'smooth', 'mean'], 'step__smooth__smoothing': [1., 10., 100.]}, 5),
    (FeatureSelectionTransformer(('x',), method='Boruta'),
     {'step__boruta__iterations': [20, 50], 'step__groot__folds': [3, 5, 10]}, 2),
    (FeatureSelectionTransformer(('x',), method='GrootCV'),
     {'step__boruta__iterations': [20, 50], 'step__groot__folds': [3, 5, 10]}, 3),
])
def test_other_method_bundles_have_conditional_counts(estimator, choices, expected):
    rows = discover(Pipeline([('step', estimator)]))
    assert compile_spec(rows, drafts_for(rows, choices))['combinations'] == expected
