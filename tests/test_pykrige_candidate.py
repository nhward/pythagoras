"""PyKrige's reviewed moving-window recipe and current-data restrictions."""
import numpy as np
import pandas as pd
import pytest
from cards.cand_methods import _analyze
from estimator_catalogue import installed_versions, load_catalogue
from evaluation_plan import build_plan
from model_commit import build_model
from proxy_data import proxy_data
from roles import RoleMap
from sklearn.base import clone

pytestmark = pytest.mark.unit
ID = 'pykrige.rk.Krige:ordinary_2d_500'


def source(n=780, dimensions=2, missing=False):
    rng = np.random.default_rng(1729)
    frame = pd.DataFrame(rng.uniform(size=(n, dimensions)), columns=[f'c{i}' for i in range(dimensions)])
    frame['target'] = frame.c0 + np.sin(frame.c1 * 3) + rng.normal(0, .1, n)
    if missing:
        frame.loc[0, 'c0'] = np.nan
    import geopandas as gpd
    predictors = list(frame.columns[:-1])
    frame = gpd.GeoDataFrame(frame, geometry=gpd.points_from_xy(frame.c0, frame.c1))
    data = proxy_data(frame, _roles=RoleMap.from_primitive({
        'predictor': predictors, 'target': ['target'], 'geometry': ['geometry']}))
    return data.with_evaluation_plan(build_plan(data, strategy='kfold', objective='independent', folds=3), card='obs_resample')


def row(data):
    return next(r for r in _analyze(data, load_catalogue(), installed_versions())['rows'] if r['id'] == ID)


def test_metadata_and_recorded_probe():
    catalogue = load_catalogue()
    profile = next(p for p in catalogue['profiles'] if p['id'] == ID)
    assert profile['validation'] == 'passed'
    assert profile['version_match']
    assert profile['validation_details']['warnings'] == []
    assert profile['validation_details']['resource_parameter_overrides'] == {}
    assert 'pipeline_three_fold_held_out_prediction' in profile['validation_details']['operations']
    assert profile['outputs'] == ['point_prediction']
    assert not profile['capabilities']['sample_weight_fit']


def test_committed_pipeline_predicts_with_500_neighbours():
    pytest.importorskip('pykrige')
    data = source()
    candidate = row(data)
    assert candidate['selectable'], candidate['reasons']
    model = build_model(candidate, data)
    assert model.named_steps['estimator'].regressor.n_closest_points == 500
    assert model.named_steps['estimator'].func is None  # Identity target transformation.
    committed = data.with_estimator(clone(model), label='Ordinary kriging', recipe_id=ID)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        fitted = committed.pipeline_for_training().fit(data.clean_frame.iloc[:600], data.clean_frame.target.iloc[:600])
        predictions = fitted.predict(data.clean_frame.iloc[600:610])
    assert len(predictions) == 10 and np.isfinite(predictions).all()
    assert data.frame.equals(committed.frame)


@pytest.mark.parametrize('kwargs, reason', [
    ({'dimensions': 3}, 'two_numeric_coordinate_predictors'),
    ({'n': 600}, 'at_least_500_training_rows_per_split'),
    ({'missing': True}, 'missing_numeric_predictors'),
])
def test_unsuitable_data_cannot_select_recipe(kwargs, reason):
    candidate = row(source(**kwargs))
    assert not candidate['selectable']
    assert any(reason in note for note in candidate['reasons'])


def test_spatial_only_not_general_tabular_or_temporal():
    from estimator_catalogue import assess_profiles
    catalogue = load_catalogue()
    for structure in ('cross_sectional', 'temporal'):
        rows = assess_profiles(catalogue, task='regression', structure=[structure],
            available_requirements=['two_numeric_coordinate_predictors', 'at_least_500_training_rows_per_split'],
            package_versions=installed_versions())
        candidate = next(r for r in rows if r['id'] == ID)
        assert candidate['profile']['structure'] == ['spatial']
        assert not candidate['selectable']
