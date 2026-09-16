"""Explicit periods, cyclic schema and composable sine/cosine encoding."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from cyclic_pandas import as_cyclic
from CyclicEncodingTransformer import CyclicEncodingTransformer
from cards import var_encode as card
from proxy_data import proxy_data
from roles import Role,RoleMap
pytestmark=pytest.mark.unit


def frame():
    data=pd.DataFrame({'hour':as_cyclic(pd.Series([0.,6.,12.,18.,None]),period=24),
        'season':as_cyclic(pd.Series(pd.Categorical(['winter','spring','summer','autumn',None],
            categories=['winter','spring','summer','autumn'],ordered=True))), 'n':[1.,2.,3.,4.,5.]})
    data.index=[0,0,1,2,3]
    return data


def test_numeric_and_categorical_cycles_have_expected_coordinates():
    X=frame();m=CyclicEncodingTransformer(['hour','season']);out=m.fit_transform(X)
    expected=np.array([[0,1],[1,0],[0,-1],[-1,0],[np.nan,np.nan]])
    np.testing.assert_allclose(out[['hour__sin','hour__cos']],expected,atol=1e-14,equal_nan=True)
    np.testing.assert_allclose(out[['season__sin','season__cos']],expected,atol=1e-14,equal_nan=True)
    assert out.index.equals(X.index)
    assert [r['Period'] for r in m.summary_]==[24,4]
    assert m.summary_[1]['Origin']=='winter'
    assert m.summary_[1]['Cycle order']=='winter → spring → summer → autumn'
    np.testing.assert_allclose(out.hour__sin.iloc[:4]**2+out.hour__cos.iloc[:4]**2,1)


def test_wraparound_negative_values_and_large_finite_values():
    m=CyclicEncodingTransformer(['hour']).fit(frame())
    incoming=pd.DataFrame({'hour':[-6.,18.,24.,0.,30.,6.,1e300]})
    out=m.transform(incoming)
    for left,right in [(0,1),(2,3),(4,5)]:
        np.testing.assert_allclose(out.iloc[left],out.iloc[right],atol=1e-14)
    assert np.isfinite(out.to_numpy()).all()
    nearby=m.transform(pd.DataFrame({'hour':[23.99,.01,12.]})).to_numpy()
    assert np.linalg.norm(nearby[0]-nearby[1]) < np.linalg.norm(nearby[0]-nearby[2])


def test_period_not_inferred_from_observed_range_or_fold():
    X=frame().iloc[:1]
    m=CyclicEncodingTransformer(['hour','season']).fit(X)
    assert [r['Period'] for r in m.summary_]==[24,4]
    out=m.transform(pd.DataFrame({'hour':[18.],'season':['autumn']}))
    np.testing.assert_allclose(out,[[-1,0,-1,0]],atol=1e-14)
    assert not hasattr(clone(m),'encodings_')


def test_missing_nonfinite_unknown_and_empty_batches():
    m=CyclicEncodingTransformer(['hour','season']).fit(frame())
    out=m.transform(pd.DataFrame({'hour':[None,np.inf,-np.inf],'season':[None,'alien',None]}))
    assert out.isna().all().all()
    strict=clone(m).set_params(handle_unknown='error').fit(frame())
    with pytest.raises(ValueError,match='Unknown cyclic'):
        strict.transform(pd.DataFrame({'hour':[0.],'season':['alien']}))
    assert strict.transform(pd.DataFrame({'hour':[None],'season':[None]})).isna().all().all()
    assert m.transform(frame().iloc[:0]).shape==(0,5)


def test_changed_cyclic_schema_is_rejected_until_refit():
    m=CyclicEncodingTransformer(['hour']).fit(frame())
    wrong=pd.DataFrame({'hour':as_cyclic(pd.Series([6.]),period=12)})
    with pytest.raises(ValueError,match='schema changed'):
        m.transform(wrong)
    m=CyclicEncodingTransformer(['season']).fit(frame())
    wrong=pd.DataFrame({'season':as_cyclic(pd.Series(pd.Categorical(['spring'],
        categories=['spring','summer','autumn','winter'],ordered=True)))})
    with pytest.raises(ValueError,match='schema changed'):
        m.transform(wrong)


def test_noncyclic_fit_rejected_and_fractional_period_supported():
    with pytest.raises(ValueError,match='explicit Cyclic dtype'):
        CyclicEncodingTransformer(['x']).fit(pd.DataFrame({'x':[1.,2.]}))
    X=pd.DataFrame({'x':as_cyclic(pd.Series([0.,.625,1.25]),period=2.5)})
    np.testing.assert_allclose(CyclicEncodingTransformer(['x']).fit_transform(X),[[0,1],[1,0],[0,-1]],atol=1e-14)


def test_pipeline_roles_removals_retention_and_collision_names():
    X=frame();X['hour__sin']=123.
    roles=RoleMap()
    for c in X:roles.set_roles(c,[Role.PREDICTOR])
    roles.set_roles('season',[Role.STRATIFIER])
    s=proxy_data(_df=X,_roles=roles,_cluster_count=3);before=s.clone()
    assert card._cyclic_predictors(s)==['hour']
    r=card._analyze_cyclic(s)
    assert not r.error
    assert r.transformer.output_columns_==['hour__sin_2','hour__cos']
    out=card._apply(s,r)
    assert out.pipeline_steps==('var_encode_cyclic',)
    assert out.cluster_count==3 and 'hour' not in out.columns
    assert out.role_map.roles_for('hour')==set()
    assert out.role_map.roles_for('hour__cos')=={Role.PREDICTOR}
    assert out.role_map.roles_for('season')=={Role.STRATIFIER}
    assert s.equals(before)
    pd.testing.assert_frame_equal(out.clean_frame,s.frame)
    pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)
    keep=card._apply(s,card._analyze_cyclic(s,remove_original=False))
    assert 'hour' in keep.columns
    pd.testing.assert_frame_equal(keep.clone().frame,keep.frame)


def test_empty_panel_is_benign_without_target():
    s=proxy_data(pd.DataFrame({'n':[1.,2.]}))
    r=card._analyze_cyclic(s)
    assert not r.error and r.transformer is None and card._apply(s,r) is s


def test_all_four_encodings_compose():
    X=pd.DataFrame({'nominal':pd.Categorical(['a','b']*6),
        'code':pd.Series(['x','y','z']*4,dtype='string'),
        'ordered':pd.Categorical(['low','medium','high']*4,categories=['low','medium','high'],ordered=True),
        'hour':as_cyclic(pd.Series(np.arange(12,dtype=float)),period=24),'target':np.arange(12,dtype=float)})
    roles=RoleMap()
    for c in X:roles.set_roles(c,[Role.TARGET if c=='target' else Role.PREDICTOR])
    s=proxy_data(_df=X,_roles=roles)
    options={'selected':('nominal','code','ordered','cyclic')}
    results=card._analyze_panels(s,options);out=s
    for kind in options['selected']:
        assert not results[kind].error
        out=card._apply(out,results[kind])
    assert out.pipeline_steps==('var_encode','var_encode_code','var_encode_ordered','var_encode_cyclic')
    pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)
