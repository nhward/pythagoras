"""Training isolation and new-observation membership behavior."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ClusterMembershipTransformer import ClusterMembershipTransformer
from proxy_data import proxy_data
from roles import Role, RoleMap

pytestmark = pytest.mark.unit

@pytest.fixture(autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


def frame():
    return pd.DataFrame({'x': [-5., -4.8, -5.1, 5., 4.9, 5.2], 'y': [-3., -3.1, -2.9, 3., 3.2, 2.8],
                         'weight': [1., 2., 3., 1., 2., 3.], 'target': [0, 1, 0, 1, 0, 1]}, index=[0]*6)

@pytest.mark.parametrize('method,centre', [('Partition','centroids'), ('Partition','medoids'), ('Mixture','centroids')])
def test_fitted_assignment_is_batch_independent_and_nominal(method, centre):
    X = frame()
    model = ClusterMembershipTransformer(('x','y'), method=method, centre=centre).fit(X)
    test = pd.DataFrame({'x': [-4.9, 5.1, np.nan], 'y': [-3., 3., 1.], 'target': [99,99,99]})
    before = model.mean_.copy()
    transformed = model.transform(test)
    assert transformed.cluster.iloc[0] != transformed.cluster.iloc[1]
    assert pd.isna(transformed.cluster.iloc[2])
    assert list(transformed.cluster.cat.categories) == ["c1", "c2"]
    assert not transformed.cluster.cat.ordered
    assert transformed.cluster.iloc[0] == model.transform(test.iloc[:1]).cluster.iloc[0]
    np.testing.assert_array_equal(before, model.mean_)
    pd.testing.assert_frame_equal(transformed.drop(columns='cluster'), test)
    assert not hasattr(clone(model), 'label_map_')
    assert model.get_feature_names_out()[-1] == 'cluster'

@pytest.mark.parametrize('centre', ['centroids','medoids'])
def test_importance_fits_but_is_not_required_to_predict(centre):
    X = frame()
    X.iloc[0, X.columns.get_loc('weight')] = 0
    model = ClusterMembershipTransformer(('x','y','weight'), weighting='weight', centre=centre).fit(X)
    assert 'weight' not in model.predictors_
    assert 0 not in model.fit_positions_
    assert model.transform(X.drop(columns='weight')).cluster.notna().all()
    scaled = X.copy(); scaled['weight'] *= 100
    other = clone(model).fit(scaled)
    pd.testing.assert_series_equal(model.transform(X).cluster, other.transform(X).cluster)


def test_mixture_rejects_unequal_weights():
    with pytest.raises(ValueError, match='unequal'):
        ClusterMembershipTransformer(('x','y'), method='Mixture', weighting='weight').fit(frame())


def test_training_subset_does_not_inherit_preview_parameters():
    X = frame()
    X.loc[:, 'x'] = [-10., -9., -8., 8., 9., 1000.]
    preview = ClusterMembershipTransformer(('x','y')).fit(X)
    train = X.iloc[:5]
    fitted = clone(preview).fit(train)
    assert fitted.scale_[0] == 10
    assert preview.scale_[0] == 1000
    assert len(fitted.fit_positions_) == 5
    fitted.transform(X.iloc[5:])
    assert fitted.scale_[0] == 10


def test_pipeline_appends_feature_preserving_clean_input_and_unfitted_recipe():
    X = frame()[['x','y']]
    source = proxy_data(_df=X, _cluster_count=2)
    scale = StandardScaler().set_output(transform='pandas')
    first = source.with_pipeline_step(scale, name='scale', preview_frame=scale.fit_transform(X))
    transformer = ClusterMembershipTransformer(('x','y'))
    preview = transformer.fit_transform(first.frame)
    roles = RoleMap(); roles.set_roles('cluster', [Role.STRATIFIER])
    result = first.with_pipeline_step(transformer, name='cluster_membership', preview_frame=preview, added_roles=roles)
    assert result.pipeline_steps == ('scale','cluster_membership')
    assert result.cluster_count == 2
    assert result.role_map.roles_for('cluster') == {Role.STRATIFIER}
    pd.testing.assert_frame_equal(result.clean_frame, X)
    assert not hasattr(result.pipeline.named_steps['cluster_membership'], 'model_')
    training = result.pipeline_for_training().fit(X.iloc[:5])
    assert 'cluster' in training.transform(X.iloc[5:])
    assert result.processing_records[-1].stage == 'Learning'
    assert result.clone().frame.cluster.dtype == preview.cluster.dtype


def test_column_additions_must_be_declared_and_cannot_overwrite():
    source = proxy_data(_df=frame()[['x','y']])
    t = ClusterMembershipTransformer(('x','y'))
    preview = t.fit_transform(source.frame)
    with pytest.raises(ValueError, match='declared'):
        source.with_pipeline_step(t, name='cluster', preview_frame=preview)
    roles = RoleMap(); roles.set_roles('x', [Role.PREDICTOR])
    with pytest.raises(ValueError, match='overwrite'):
        source.with_pipeline_step(t, name='cluster', preview_frame=preview, added_roles=roles)
