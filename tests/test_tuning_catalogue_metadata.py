"""The reviewed front-table scope has explicit metadata and usable scalar grids."""
import importlib
import inspect
import json
from pathlib import Path

import pytest
from estimator_catalogue import assess_profiles, load_catalogue
from sklearn.pipeline import Pipeline
from tuning import STANDARD_METADATA, compile_spec, discover, search_values, visible

pytestmark = pytest.mark.unit
REGISTRY = json.loads((Path(__file__).parents[1] / 'app/data/tuning_parameters.json').read_text())
METHODS = REGISTRY['review_scope']['methods']
# Keep validation coverage for the retained alternative after hiding it by default.
REVIEWED_METHODS = {**METHODS, 'skglm.ElasticNet': 'skglm.estimators.ElasticNet'}


def test_scope_matches_potential_front_table_regression_and_classification_methods():
    # Union across data/resampling contexts; installation and current data are not
    # permanent exclusions. Superseded/benchmark-only/unreviewed-input recipes are.
    eligible = set()
    for task in ('regressor', 'classifier'):
        for row in assess_profiles(load_catalogue(), task=task):
            if (row['purpose'] != 'benchmark_only' and not row.get('superseded_by')
                    and not row['reasons']
                    and row['input_assessment']['readiness'] == 'Meets reviewed input requirements'):
                eligible.add(row['estimator'])
    assert set(METHODS) == eligible


@pytest.mark.parametrize('public,identity', REVIEWED_METHODS.items())
def test_every_constructor_parameter_has_reviewed_metadata(public, identity):
    module, name = public.rsplit('.', 1)
    try:
        cls = getattr(importlib.import_module(module), name)
    except ImportError:
        pytest.skip('Optional estimator package is not installed')
    parameters = set(inspect.signature(cls).parameters) - {'args', 'kwargs'}
    if public.startswith('xgboost.'):
        parameters.update(cls().get_params())  # Inherited kwargs are documented too.
    metadata = STANDARD_METADATA[identity]
    assert parameters <= set(metadata)
    for entry in metadata.values():
        assert entry['purpose'] in {'recommended', 'advanced', 'operational'}
        assert entry['reason'] and entry['reference'].startswith('https://')
    rows = {r['path'].split('__', 1)[1]: r for r in discover(Pipeline([('model', cls())]))}
    assert parameters <= set(rows)
    for name in parameters:
        row = rows[name]
        assert row['purpose'] == metadata[name]['purpose']
        assert visible(row, 'everything')
        if row['fixed']:
            assert not visible(row, 'unreviewed')


@pytest.mark.parametrize('public,identity', REVIEWED_METHODS.items())
def test_suggested_scalar_values_and_installed_parameter_constraints(public, identity):
    module, name = public.rsplit('.', 1)
    try:
        cls = getattr(importlib.import_module(module), name)
    except ImportError:
        cls = None
    for parameter, metadata in STANDARD_METADATA[identity].items():
        suggested = metadata.get('suggested')
        if not suggested:
            continue  # Structured objects and scale-dependent settings have no invented grid.
        values = search_values({'kind': metadata['kind'], 'mode': 'values', **suggested}, metadata)
        assert values
        if metadata['purpose'] == 'operational' or cls is None:
            continue
        for value in values:
            estimator = cls().set_params(**{parameter: value})
            if getattr(estimator, '_parameter_constraints', None):
                estimator._validate_params()


def test_xgboost_documented_optional_kwargs_are_discoverable_and_compilable():
    cls = pytest.importorskip('xgboost').XGBRegressor
    model = cls(n_estimators=3)
    before = model.get_params().copy()
    rows = discover(Pipeline([('model', model)]))
    row = next(r for r in rows if r['path'] == 'model__rate_drop')
    assert row['purpose'] == 'advanced' and 'dart' in row['reason']
    spec = compile_spec(rows, {row['path']: dict(row['definition'], active=True, owner=row['owner'])})
    assert spec['grid'] == {'model__rate_drop': [0.0, 0.1, 0.3]}
    assert model.get_params() == before
