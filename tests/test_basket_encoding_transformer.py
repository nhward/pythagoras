import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from list_pandas import as_list
from BasketEncodingTransformer import BasketEncodingTransformer
from cards import var_encode as card
from proxy_data import proxy_data
from roles import Role, RoleMap
pytestmark=pytest.mark.unit


def test_binary_duplicates_missing_empty_and_unseen_items():
    X=pd.DataFrame({'basket':as_list(pd.Series([['a','a','b'],['b'],[],None,[None]]))})
    X.index=[0,0,2,3,4]
    model=BasketEncodingTransformer(['basket'])
    out=model.fit_transform(X)
    np.testing.assert_equal(out.to_numpy(dtype=float,na_value=np.nan),
        [[1,1],[0,1],[0,0],[np.nan,np.nan],[0,0]])
    assert out.index.equals(X.index)
    future=pd.DataFrame({'basket':[['b','new'],['new'],[],None]})
    actual=model.transform(future)
    assert actual.columns.tolist()==['basket__a','basket__b']
    np.testing.assert_equal(actual.to_numpy(dtype=float,na_value=np.nan),[[0,1],[0,0],[0,0],[np.nan,np.nan]])
    assert not hasattr(clone(model),'encodings_')
    assert model.transform(X.iloc[:0]).shape==(0,2)
    assert model.get_feature_names_out().tolist()==out.columns.tolist()


def test_no_vocabulary_retains_original_and_invalid_values_rejected():
    X=pd.DataFrame({'basket':as_list(pd.Series([[],None]))})
    out=BasketEncodingTransformer(['basket']).fit_transform(X)
    pd.testing.assert_frame_equal(out,X)
    with pytest.raises(ValueError,match='Basket dtype'):
        BasketEncodingTransformer(['basket']).fit(pd.DataFrame({'basket':['abc']}))
    with pytest.raises(ValueError,match='nested'):
        BasketEncodingTransformer(['basket']).fit(pd.DataFrame({'basket':as_list(pd.Series([[[1,2]]]))}))
    X=pd.DataFrame({'basket':as_list(pd.Series([list(range(4097))]))})
    with pytest.raises(ValueError,match='4096'):
        BasketEncodingTransformer(['basket']).fit(X)


def test_pipeline_roles_collision_names_audit_and_retention():
    X=pd.DataFrame({'basket':as_list(pd.Series([[1,'1'],['1']])),
        'stratum':as_list(pd.Series([['x'],['y']])), 'basket__1':[5,6], 'flag':[True,False]})
    roles=RoleMap()
    for c in X:roles.set_roles(c,[Role.PREDICTOR])
    roles.set_roles('stratum',[Role.STRATIFIER])
    source=proxy_data(_df=X,_roles=roles)
    assert card._basket_predictors(source)==['basket']
    selected=('logical','basket')
    results=card._analyze_panels(source,{'selected':selected})
    out=source
    for kind in selected:
        assert not results[kind].error
        out=card._apply(out,results[kind])
    assert 'basket__1_2' in out.columns and 'basket__1_3' in out.columns
    assert out.role_map.roles_for('basket__1_2')=={Role.PREDICTOR}
    assert out.role_map.roles_for('stratum')=={Role.STRATIFIER}
    pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)
    summary,table=card._encoding_audit(source,results,selected)
    assert summary['steps']==2 and summary['removed']==2
    assert table.iloc[-1]['Encoding']=='Binary item presence'
    kept=card._apply(source,card._analyze_basket(source,remove_original=False))
    assert 'basket' in kept.columns
    empty=proxy_data(pd.DataFrame({'n':[1,2]}))
    assert card._apply(empty,card._analyze_basket(empty)) is empty
