"""Supervised encoding: no self-target training leakage, roles and composition."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from TargetEncodingTransformer import TargetEncodingTransformer
from cards import var_encode as card
from proxy_data import proxy_data
from roles import Role, RoleMap

pytestmark = pytest.mark.unit


def source(kind='continuous'):
    frame=pd.DataFrame({'code':pd.Series(['a','b','c']*6,dtype='string'),
        'other':pd.Series(['id'+str(i) for i in range(18)],dtype='string'),
        'nominal':pd.Categorical(['yes','no']*9),'target':np.arange(18,dtype=float),
        'weight':np.arange(18)+1.})
    if kind=='binary':frame['target']=pd.Categorical(['negative','positive']*9)
    if kind=='multiclass':frame['target']=pd.Categorical(['one','two','three']*6)
    frame.index=[0]*18
    roles=RoleMap()
    for column in frame: roles.set_roles(column,[Role.PREDICTOR])
    roles.set_roles('target',[Role.TARGET]);roles.set_roles('other',[Role.IDENTIFIER]);roles.set_roles('weight',[Role.WEIGHTING])
    return proxy_data(_df=frame,_roles=roles,_cluster_count=3)


def options(selected=('nominal','code'),**code):
    return {'nominal':{},'code':code,'selected':selected}


@pytest.mark.parametrize('method',['auto','smooth','mean'])
def test_all_methods_cross_fit_and_pass_through(method):
    s=source();before=s.clone();result=card._analyze_code(s,method=method)
    assert not result.error
    assert result.transformer.columns==['code']
    assert result.frame.index.equals(s.frame.index)
    assert result.frame.shape==s.frame.shape
    pd.testing.assert_series_equal(result.frame.target,s.frame.target)
    assert np.isfinite(result.frame.code__target).all()
    assert s.equals(before)


def test_own_outcome_does_not_affect_own_training_encoding():
    X=pd.DataFrame({'code':pd.Series(['a']*12,dtype='string'),'target':np.arange(12,dtype=float)})
    for method in ('auto','smooth','mean'):
        a=TargetEncodingTransformer(['code'],target='target',method=method,cv=3).fit_transform(X)
        changed=X.copy();changed.loc[3,'target']=10000.
        b=TargetEncodingTransformer(['code'],target='target',method=method,cv=3).fit_transform(changed)
        assert a.code__target.iloc[3]==pytest.approx(b.code__target.iloc[3])
        assert a.code__target.iloc[3]!=pytest.approx(X.target.iloc[3])


def test_unique_codes_are_not_encoded_as_their_own_targets():
    X=pd.DataFrame({'code':pd.Series(['id'+str(i) for i in range(20)],dtype='string'),'target':np.arange(20,dtype=float)})
    m=TargetEncodingTransformer(['code'],target='target',method='mean',cv=5)
    training=m.fit_transform(X)
    assert not np.allclose(training.code__target,X.target)
    np.testing.assert_allclose(m.transform(X).code__target,X.target)


def test_fixed_smoothing_and_inference_never_reads_outcomes():
    X=pd.DataFrame({'code':['a','a','b','b'],'target':[1.,3.,5.,7.]})
    m=TargetEncodingTransformer(['code'],target='target',method='smooth',smoothing=2,cv=2).fit(X)
    test=pd.DataFrame({'code':['a','b','new',None]})
    np.testing.assert_allclose(m.transform(test).code__target,[3,5,4,4])
    with_target=test.assign(target=[-9999]*4)
    np.testing.assert_allclose(m.transform(with_target).code__target,[3,5,4,4])
    assert m.transform(test.iloc[:0]).shape==(0,1)


def test_explicit_y_overrides_frame_target_and_targetless_fit():
    X=pd.DataFrame({'code':['a','a','b','b'],'target':[999]*4})
    y=np.array([1.,3.,5.,7.])
    a=TargetEncodingTransformer(['code'],target='target',method='mean',cv=2).fit(X,y)
    b=clone(a).fit(X.drop(columns='target'),y)
    np.testing.assert_allclose(a.transform(X).code__target,[2,2,6,6])
    np.testing.assert_allclose(b.transform(X).code__target,[2,2,6,6])
    assert not hasattr(clone(a),'encoder_')


@pytest.mark.parametrize('kind,width',[('binary',1),('multiclass',3)])
def test_classification_probability_outputs(kind,width):
    s=source(kind);result=card._analyze_code(s)
    assert not result.error
    model=result.transformer
    assert len(model.output_columns_)==width
    values=result.frame[model.output_columns_].to_numpy()
    assert ((values>=0)&(values<=1)).all()
    if width==3:np.testing.assert_allclose(values.sum(axis=1),1)
    else:assert model.output_columns_==['code__target__positive']
    out=card._apply(s,result)
    assert all(out.role_map.roles_for(c)=={Role.PREDICTOR} for c in model.output_columns_)
    assert out.role_map.roles_for('target')=={Role.TARGET}


def test_target_validation_and_benign_missing_code():
    s=source();s.role_map.clear_roles('target')
    assert 'exactly one Target' in card._analyze_code(s).error
    s=source();s.role_map.set_roles('other',[Role.TARGET])
    assert 'More than one' in card._analyze_code(s).error
    s=source();s.frame.iloc[0,s.frame.columns.get_loc('target')]=np.nan
    assert 'missing outcomes' in card._analyze_code(s).error
    s=source();s.frame['target']=np.inf
    assert 'finite' in card._analyze_code(s).error
    s=source('binary');s.frame['target']=pd.Categorical(['rare']+['common']*17)
    assert 'two observations in every' in card._analyze_code(s).error
    s=source();s.role_map.set_roles('code',[Role.STRATIFIER])
    result=card._analyze_code(s)
    assert result.transformer is None and not result.error and card._apply(s,result) is s


def test_missing_codes_are_a_learned_category():
    X=pd.DataFrame({'code':['a','a',None,None],'target':[1.,3.,5.,7.]})
    m=TargetEncodingTransformer(['code'],target='target',method='mean',cv=2).fit(X)
    np.testing.assert_allclose(m.transform(pd.DataFrame({'code':[None,'unseen']})).code__target,[6,4])


def test_fold_limits_numeric_integer_targets_and_retained_columns():
    s=source();s.frame['target']=s.frame.target.astype(int)
    r=card._analyze_code(s,cv=10,remove_original=False)
    assert not r.error and r.transformer.target_type=='continuous'
    assert 'code' in r.frame and r.transformer.removed_columns_==[]
    s=source('binary');r=card._analyze_code(s,cv=10)
    assert r.transformer.cv_==9


class CaptureTraining(BaseEstimator):
    def fit(self,X,y=None):
        self.seen_=X.copy()
        return self


def test_estimator_pipeline_uses_cross_fitted_training_values():
    X=pd.DataFrame({'code':['a']*12,'target':np.arange(12,dtype=float)})
    encoder=TargetEncodingTransformer(['code'],target='target',cv=3)
    pipeline=Pipeline([('encoder',encoder),('model',CaptureTraining())]).fit(X,X.target)
    expected=clone(encoder).fit_transform(X,X.target)
    pd.testing.assert_frame_equal(pipeline.named_steps['model'].seen_,expected)


def test_both_panels_compose_and_clones_do_not_keep_learned_mappings():
    s=source();before=s.clone();results=card._analyze_panels(s,options())
    assert not results['code'].error
    out=card._apply(card._apply(s,results['nominal']),results['code'])
    assert out.pipeline_steps==('var_encode','var_encode_code')
    assert 'code' not in out.columns and 'nominal' not in out.columns
    assert s.equals(before) and out.cluster_count==3
    pd.testing.assert_frame_equal(out.clean_frame,s.frame)
    pd.testing.assert_frame_equal(out.clone().frame,out.frame)
    recipe=out.pipeline_for_training()
    assert not hasattr(recipe.steps[-1][1],'encoder_')
    pd.testing.assert_frame_equal(recipe.fit_transform(out.clean_frame),out.frame)
    train=s.frame.iloc[:12];test=s.frame.iloc[12:].drop(columns='target')
    held=out.pipeline_for_training().fit(train).transform(test)
    assert 'target' not in held and np.isfinite(held.code__target).all()


def test_collision_between_two_panels_is_safe():
    s=source()
    frame=s.frame.rename(columns={'code':'a__b','nominal':'a'})
    frame['a']=pd.Categorical(['b__target','other']*9)
    roles=RoleMap()
    for c in frame:roles.set_roles(c,[Role.TARGET if c=='target' else Role.PREDICTOR])
    roles.set_roles('other',[Role.IDENTIFIER]);roles.set_roles('weight',[Role.WEIGHTING])
    s=proxy_data(_df=frame,_roles=roles)
    results=card._analyze_panels(s,options())
    assert not results['code'].error
    out=card._apply(card._apply(s,results['nominal']),results['code'])
    assert out.frame.columns.is_unique
    pd.testing.assert_frame_equal(out.pipeline_for_training().fit_transform(out.clean_frame),out.frame)


def test_weights_do_not_change_encodings():
    s=source();a=card._analyze_code(s)
    s.frame['weight']*=100
    b=card._analyze_code(s)
    pd.testing.assert_frame_equal(a.frame[a.transformer.output_columns_],b.frame[b.transformer.output_columns_])
