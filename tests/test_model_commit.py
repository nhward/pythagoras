import numpy as np
import pandas as pd
import pytest
from cards.data_models import _analyze
from estimator_catalogue import installed_versions, load_catalogue
from model_commit import build_model
from pandas.testing import assert_frame_equal
from proxy_data import proxy_data
from roles import RoleMap
from sklearn.preprocessing import FunctionTransformer

pytestmark = pytest.mark.unit


def fixture():
    frame = pd.DataFrame({'x': [1., 2., 3., 4.], 'y': [2., 4., 6., 8.]})
    source = proxy_data(frame, _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
    rows = _analyze(source, load_catalogue(), installed_versions())['rows']
    row = next(r for r in rows if r['estimator'] == 'sklearn.linear_model.Ridge'
               and r['profile']['construction']['preparation'] == 'impute_onehot')
    return source, row


def test_commit_clones_preserves_data_replaces_model_and_fits_without_target_leakage():
    source, row = fixture()
    source = source.with_pipeline_step(FunctionTransformer(), name='upstream', preview_frame=source.frame)
    model = build_model(row, source.model_predictors)
    committed = source.with_estimator(model, label='Linear Regression', recipe_id=row['id'])
    assert committed.pipeline_steps == ('upstream', 'data_models')
    assert_frame_equal(committed.frame, source.frame)
    assert_frame_equal(committed.clean_frame, source.clean_frame)
    assert not hasattr(committed.pipeline.named_steps.data_models.named_steps.estimator, 'coef_')
    trained = committed.pipeline_for_training().fit(source.clean_frame, source.frame.y)
    assert np.isfinite(trained.predict(source.clean_frame)).all()
    assert trained.named_steps.data_models.named_steps.estimator.n_features_in_ == 1
    again = committed.with_estimator(model, label='Linear Regression', recipe_id=row['id'])
    assert again.pipeline_steps == committed.pipeline_steps
    assert len([r for r in again.processing_records if r.card == 'data_models']) == 1
    assert again.processing_records[-1].operation == 'Committed estimator: Linear Regression'
    after = again.with_pipeline_step(FunctionTransformer(), name='later', preview_frame=source.frame)
    assert after.pipeline_steps == ('upstream', 'later', 'data_models')


def test_unvalidated_recipe_is_not_constructed():
    source, row = fixture()
    with pytest.raises(ValueError, match='not validated'):
        build_model(dict(row, selectable=False), source.model_predictors)
