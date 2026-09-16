import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from LogicalEncodingTransformer import LogicalEncodingTransformer
from cards import var_encode as card
from proxy_data import proxy_data
from roles import Role, RoleMap
pytestmark = pytest.mark.unit


def test_nullable_integers_missing_duplicate_index_and_inference():
    X = pd.DataFrame({'flag':pd.Series([True,False,None],dtype='boolean'), 'n':[3,4,5]})
    X.index = [1,1,2]
    model = LogicalEncodingTransformer(['flag'])
    out = model.fit_transform(X)
    pd.testing.assert_series_equal(out.flag__integer,
        pd.Series([1,0,None],index=X.index,dtype='Int64',name='flag__integer'))
    assert model.get_feature_names_out().tolist() == ['n','flag__integer']
    assert not hasattr(clone(model),'encodings_')
    incoming = pd.DataFrame({'flag':pd.Series([np.bool_(False),None,True],dtype=object)})
    assert model.transform(incoming).flag__integer.tolist() == [0,pd.NA,1]
    assert model.transform(X.iloc[:0]).empty
    for bad in ['False',1,0]:
        with pytest.raises(ValueError,match='boolean values'):
            model.transform(pd.DataFrame({'flag':[bad]}))
    with pytest.raises(ValueError,match='boolean.*dtype'):
        model.fit(pd.DataFrame({'flag':[0,1]}))


def test_constant_and_all_missing_booleans():
    X = pd.DataFrame({'yes':[True,True], 'missing':pd.Series([None,None],dtype='boolean')})
    out = LogicalEncodingTransformer(['yes','missing']).fit_transform(X)
    assert out.yes__integer.tolist() == [1,1]
    assert out.missing__integer.isna().all()


def test_roles_collisions_retention_and_unfitted_pipeline():
    X = pd.DataFrame({'flag':[True,False], 'stratum':[False,True], 'target':[True,False],
        'flag__integer':[9,8]})
    roles = RoleMap()
    for c in X: roles.set_roles(c,[Role.PREDICTOR])
    roles.set_roles('stratum',[Role.STRATIFIER]); roles.set_roles('target',[Role.TARGET])
    source = proxy_data(_df=X,_roles=roles)
    assert card._logical_predictors(source) == ['flag']
    result = card._analyze_logical(source)
    out = card._apply(source,result)
    assert out.pipeline_steps == ('var_encode_logical',)
    assert out.role_map.roles_for('flag__integer_2') == {Role.PREDICTOR}
    assert out.role_map.roles_for('stratum') == {Role.STRATIFIER}
    assert out.role_map.roles_for('flag') == set()
    pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)
    keep = card._apply(source,card._analyze_logical(source,remove_original=False))
    assert keep.frame.flag.equals(source.frame.flag)
    empty = proxy_data(pd.DataFrame({'n':[1,2]}))
    assert card._apply(empty,card._analyze_logical(empty)) is empty


from cyclic_pandas import as_cyclic

def test_all_five_encodings_compose():
    X=pd.DataFrame({'nominal':pd.Categorical(['a','b']*6),
        'code':pd.Series(['x','y','z']*4,dtype='string'),
        'ordered':pd.Categorical(['low','medium','high']*4,categories=['low','medium','high'],ordered=True),
        'hour':as_cyclic(pd.Series(np.arange(12,dtype=float)),period=24),'target':np.arange(12,dtype=float)})
    X['flag']=pd.Series([True,False,None]*4,dtype='boolean')
    roles=RoleMap()
    for c in X:roles.set_roles(c,[Role.TARGET if c=='target' else Role.PREDICTOR])
    s=proxy_data(_df=X,_roles=roles)
    options={'selected':('nominal','code','ordered','cyclic','logical')}
    results=card._analyze_panels(s,options);out=s
    for kind in options['selected']:
        assert not results[kind].error
        out=card._apply(out,results[kind])
    assert out.pipeline_steps==('var_encode','var_encode_code','var_encode_ordered','var_encode_cyclic','var_encode_logical')
    pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)
