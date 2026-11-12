"""One exported step and one K, with independently fitted method models."""
import numpy as np
import pandas as pd
import pytest
from ClusterMembershipsTransformer import ClusterMembershipsTransformer
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from tuning import discover

pytestmark = pytest.mark.unit


def frame():
    rng = np.random.default_rng(52)
    return pd.DataFrame({'x': np.r_[rng.normal(-4, .1, 15), rng.normal(4, .1, 15)],
                         'y': rng.normal(size=30)}, index=[0] * 30)


def test_one_k_all_methods_clone_and_inference_rows():
    X = frame()
    transformer = ClusterMembershipsTransformer(('x', 'y'), methods=('Partition', 'Mixture', 'Density'), density_min_points=3)
    params = transformer.get_params(deep=True)
    assert [p for p in params if p.endswith('n_clusters')] == ['n_clusters']
    assert params['density_min_points'] == 3 and params['density_border'] is True
    fitted = clone(transformer).set_params(n_clusters=3).fit(X.iloc[:24])
    assert all(model.n_clusters == 3 and len(model.fit_positions_) == 24 for model in fitted.models_.values())
    assert all(model.predictors_ == ['x', 'y'] for model in fitted.models_.values())
    test = X.iloc[24:].copy()
    test.iloc[0, 0] = np.nan
    result = fitted.transform(test)
    assert result.index.equals(test.index)
    assert len(result) == len(test)
    for method in fitted.methods:
        column = f'cluster__{method.lower()}'
        assert result[column].isna().iloc[0]
        assert isinstance(result[column].dtype, pd.CategoricalDtype)
    assert list(fitted.get_feature_names_out()) == list(result.columns)
    assert not hasattr(clone(fitted), 'models_')
    rows = discover(Pipeline([('obs_clusters', transformer)]))
    tunable = [r['path'] for r in rows if not r['fixed']]
    assert tunable == ['obs_clusters__n_clusters']
    assert not any('__Partition__' in r['path'] for r in rows)


def test_method_specific_parameters_can_be_configured_without_another_k():
    transformer = ClusterMembershipsTransformer(('x', 'y'), methods=('Density',))
    fitted = clone(transformer).set_params(density_min_points=3, density_border=False).fit(frame())
    density = fitted.models_['Density']
    assert density.min_points == 3 and density.border is False
    assert density.n_clusters == fitted.n_clusters


@pytest.mark.parametrize('kwargs', [{'methods': ()}, {'methods': ('Partition', 'Partition')},
    {'methods': ('Other',)}, {'output_columns': (('Partition', 'x'),)},
    {'methods': ('Partition', 'Mixture'), 'output_columns': (('Partition', 'c'), ('Mixture', 'c'))}])
def test_invalid_group_or_output_schema_rejected(kwargs):
    with pytest.raises(ValueError): ClusterMembershipsTransformer(('x',), **kwargs).fit(frame())
