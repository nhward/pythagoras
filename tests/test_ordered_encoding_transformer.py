"""Ordered schema preservation, sklearn ranks and R-style polynomial contrasts."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.preprocessing import OrdinalEncoder
from OrderedEncodingTransformer import OrderedEncodingTransformer,polynomial_contrasts
from cards import var_encode as card
from proxy_data import proxy_data
from roles import Role,RoleMap

pytestmark=pytest.mark.unit


def frame():
    return pd.DataFrame({'size':pd.Categorical(['large','small','medium',None,'large'],
        categories=['small','medium','large','huge'],ordered=True),
        'number':[1.,2.,3.,4.,5.]},index=[0,0,1,2,3])


def source():
    data=frame()
    data['unordered']=pd.Categorical(['a','b','a','b','a'])
    data['target']=pd.Categorical(['x','y','x','y','x'],ordered=True)
    roles=RoleMap()
    for c in data:roles.set_roles(c,[Role.TARGET if c=='target' else Role.PREDICTOR])
    return proxy_data(_df=data,_roles=roles)


def test_rank_order_and_unobserved_declared_levels():
    X=frame();model=OrderedEncodingTransformer(['size']).fit(X)
    assert isinstance(model.encodings_['size'][1],OrdinalEncoder)
    np.testing.assert_allclose(model.transform(X).size__rank,[2,0,1,np.nan,2],equal_nan=True)
    assert model.summary_[0]['Cardinality']==4 and model.summary_[0]['Observed levels']==3
    assert model.transform(pd.DataFrame({'size':['huge']})).size__rank.iloc[0]==3
    assert model.transform(X).index.equals(X.index)


def test_numeric_labels_need_not_be_numerically_sorted():
    X=pd.DataFrame({'x':pd.Categorical([2,10,1],categories=[10,1,2],ordered=True)})
    actual=OrderedEncodingTransformer(['x']).fit_transform(X)
    np.testing.assert_allclose(actual.x__rank,[2,0,1])


@pytest.mark.parametrize('method',['ordinal','polynomial'])
def test_missing_unknown_and_targetless_inference(method):
    model=OrderedEncodingTransformer(['size'],method=method).fit(frame())
    result=model.transform(pd.DataFrame({'size':['new',None,'medium']}))
    assert result.iloc[:2].isna().all().all()
    assert result.iloc[2].notna().all()
    strict=clone(model).set_params(handle_unknown='error').fit(frame())
    with pytest.raises(ValueError,match='unknown'):
        strict.transform(pd.DataFrame({'size':['new']}))
    assert strict.transform(pd.DataFrame({'size':[None]})).isna().all().all()
    assert model.transform(frame().iloc[:0]).shape[0]==0


def test_r_contr_poly_four_level_coefficients():
    expected=np.column_stack([np.array([-3,-1,1,3])/np.sqrt(20),
        np.array([1,-1,-1,1])/2,np.array([-1,3,-3,1])/np.sqrt(20)])
    np.testing.assert_allclose(polynomial_contrasts(4),expected,atol=1e-14)


@pytest.mark.parametrize('levels,degree',[(2,None),(3,None),(12,None),(65,64),(100,8)])
def test_polynomials_are_orthonormal_over_levels(levels,degree):
    matrix=polynomial_contrasts(levels,degree)
    np.testing.assert_allclose(matrix.sum(axis=0),0,atol=1e-12)
    np.testing.assert_allclose(matrix.T@matrix,np.eye(matrix.shape[1]),atol=1e-12)
    assert matrix.shape==(levels,min(degree or levels-1,levels-1))
    assert np.all(np.diff(matrix[:,0])>0)


def test_polynomial_degree_and_collision_safe_names():
    X=frame();X['size__L']=10.
    model=OrderedEncodingTransformer(['size'],method='polynomial',degree=2).fit(X)
    assert model.output_columns_==['size__L_2','size__Q']
    assert model.transform(X).size__L.eq(10).all()
    assert not hasattr(clone(model),'encodings_')
    X=pd.DataFrame({'x':pd.Categorical([0],categories=list(range(66)),ordered=True)})
    with pytest.raises(ValueError,match='at most 64'):
        OrderedEncodingTransformer(['x'],method='polynomial').fit(X)


def test_role_filter_retention_and_pipeline_training_schema():
    s=source();before=s.clone()
    assert card._ordered_predictors(s)==['size']
    result=card._analyze_ordered(s,method='polynomial')
    assert not result.error
    out=card._apply(s,result)
    assert out.pipeline_steps==('var_encode_ordered',)
    assert 'size' not in out.columns and out.role_map.roles_for('size')==set()
    assert all(out.role_map.roles_for(c)=={Role.PREDICTOR} for c in result.transformer.output_columns_)
    assert out.role_map.roles_for('target')=={Role.TARGET}
    assert s.equals(before)
    pd.testing.assert_frame_equal(out.clean_frame,s.frame)
    pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)
    # A fold containing only one observed level still retains the declared order.
    fitted=out.pipeline_for_training().fit(s.frame.iloc[:1])
    encoded=fitted.transform(s.frame.iloc[1:])
    pd.testing.assert_frame_equal(encoded,out.frame.iloc[1:])
    keep=card._apply(s,card._analyze_ordered(s,remove_original=False))
    assert 'size' in keep.columns and keep.role_map.roles_for('size')=={Role.PREDICTOR}


def test_empty_or_degenerate_panel_is_benign():
    s=proxy_data(pd.DataFrame({'n':[1,2]}))
    result=card._analyze_ordered(s)
    assert not result.error and card._apply(s,result) is s
    s=proxy_data(pd.DataFrame({'x':pd.Categorical(['a','a'],ordered=True)}))
    result=card._analyze_ordered(s)
    assert not result.error and card._apply(s,result) is s


def test_all_three_panels_compose_without_target_role_changes():
    frame=pd.DataFrame({'nominal':pd.Categorical(['a','b']*6),
        'code':pd.Series(['x','y','z']*4,dtype='string'),
        'ordered':pd.Categorical(['low','medium','high']*4,categories=['low','medium','high'],ordered=True),
        'target':np.arange(12,dtype=float)})
    roles=RoleMap()
    for c in frame:roles.set_roles(c,[Role.TARGET if c=='target' else Role.PREDICTOR])
    s=proxy_data(_df=frame,_roles=roles)
    options={'nominal':{},'code':{},'ordered':{'method':'polynomial'},'selected':('nominal','code','ordered')}
    analyses=card._analyze_panels(s,options)
    out=s
    for kind in options['selected']:
        assert not analyses[kind].error
        out=card._apply(out,analyses[kind])
    assert out.pipeline_steps==('var_encode','var_encode_code','var_encode_ordered')
    assert out.role_map.roles_for('target')=={Role.TARGET}
    pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)
